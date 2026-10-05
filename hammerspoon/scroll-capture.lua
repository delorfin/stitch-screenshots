-- Scrolling screenshot of the frontmost window.
-- ctrl+cmd+shift+2 starts; Esc (or the hotkey again) stops early. Otherwise it
-- scrolls until the content stops moving, then stitches the frames with
-- ../stitch.py --window and saves to ~/Downloads + the clipboard.
-- Silent on success; alerts only on failure.
-- Animated content (a blinking caret, video) never settles: stop it with Esc.

local M = {}

local CONTENT_ONLY = false  -- true: drop the window's fixed UI, keep only the scrolled content
local STEP_FRAC = 1 / 3   -- scroll step, as a fraction of the window height
local CHUNK = 40          -- px per synthetic scroll event (some apps clamp big deltas)
local TICK = 0.016        -- seconds between scroll events, like a trackpad
local HOLD = 0.1          -- pause before "lifting the finger", so nothing flings
local POLL = 0.15         -- seconds between captures while waiting for the view to settle
local MAX_WAIT = 3        -- seconds to wait for a frame to settle before taking it as is
local ANIMATED_WAIT = 1.2 -- once a frame never settled (an animation), wait only this long

local log = hs.logger.new("scrollcap", "info")

local here = hs.fs.pathToAbsolute(debug.getinfo(1, "S").source:sub(2)):match("(.*)/")
local STITCH = here .. "/../stitch.py"
-- login shell, so python3 is the user's; last line only (shell startup may print)
local PYTHON = (hs.execute("command -v python3", true) or ""):match("([^\n]+)%s*$") or ""

local running, stopRequested = false, false
local session = nil       -- {dir, origMouse, escKey} while a capture runs
local tasks = {}          -- keep hs.task objects alive until their callbacks fire

local function readFile(path)
    local f = io.open(path, "rb")
    if not f then return nil end
    local data = f:read("a")
    f:close()
    return data
end

local function removeDir(dir)
    if not hs.fs.attributes(dir) then return end
    for file in hs.fs.dir(dir) do
        if file ~= "." and file ~= ".." then os.remove(dir .. "/" .. file) end
    end
    hs.fs.rmdir(dir)
end

-- Frames are deleted after each capture; a failed one keeps them for a look.
-- Those leftovers go away a day later, on the next capture.
local function removeOldCaptures()
    local tmp = os.getenv("TMPDIR")
    local cutoff = os.time() - 24 * 3600
    for name in hs.fs.dir(tmp) do
        if name:match("^scroll%-capture%-") then
            local attrs = hs.fs.attributes(tmp .. name)
            if attrs and attrs.mode == "directory" and attrs.modification < cutoff then
                removeDir(tmp .. name)
            end
        end
    end
end

local function runTask(cmd, args, cb)
    local t
    t = hs.task.new(cmd, function(...)
        tasks[t] = nil
        cb(...)
    end, args)
    tasks[t] = true
    if not t:start() then
        tasks[t] = nil
        return false
    end
    return true
end

-- Keep filenames readable for non-ASCII titles: drop only path-unsafe bytes,
-- and cut at a UTF-8 character boundary.
local function safeName(s, maxChars)
    s = (s or ""):gsub("%s+", "_"):gsub("[%c/:\\]", "")
    if not utf8.len(s) then return s:sub(1, maxChars) end   -- not valid UTF-8
    local cut = utf8.offset(s, maxChars + 1)
    return cut and s:sub(1, cut - 1) or s
end

local function outputPath(win)
    local app = win:application()
    local timeStamp = string.gsub(os.date("%Y-%m-%d_%T"), ":", ".")
    return os.getenv("HOME") .. "/Downloads/scroll-" .. safeName(app and app:name(), 30) .. "-"
        .. safeName(win:title(), 40) .. "-" .. timeStamp .. ".png"
end

-- Undo the capture's side effects: Esc binding, moved pointer, running flag.
local function endSession()
    if not session then return end
    if session.escKey then session.escKey:delete() end
    if session.origMouse then hs.mouse.absolutePosition(session.origMouse) end
    session, running = nil, false
end

-- Any error mid-capture: clean up, keep the frames for inspection, tell the user.
local function fail(err, dir)
    log.e(err)
    -- a late error from an earlier capture's stitch must not end a newer one
    if session and (dir == nil or session.dir == dir) then endSession() end
    hs.alert.show("Scroll capture failed; see Hammerspoon console"
        .. (dir and hs.fs.attributes(dir) and ("\nframes kept in " .. dir) or ""))
end

-- Run fn(...) and route any error to fail(), so timer/task callbacks can't
-- leave the capture wedged.
local function guarded(fn, dir)
    return function(...)
        local ok, err = xpcall(fn, debug.traceback, ...)
        if not ok then fail(err, dir) end
    end
end

local function deliver(out)
    local img = hs.image.imageFromPath(out)
    if img then hs.pasteboard.writeObjects(img) end
end

-- The vertical scrollbar of the scroll area under `point`, as image-pixel
-- columns "a:b" of the window capture, if the app reports one via
-- Accessibility (native apps do; Electron and many web views don't).
local function scrollbarCols(win, point)
    local frame = win:frame()
    local scale = win:screen():currentMode().scale or 1
    local queue, visited, best = {{hs.axuielement.windowElement(win), 0}}, 0, nil
    local deadline = hs.timer.secondsSinceEpoch() + 1.5   -- slow apps: give up, don't stall
    while #queue > 0 and visited < 2000 and hs.timer.secondsSinceEpoch() < deadline do
        local el, depth = table.unpack(table.remove(queue, 1))
        visited = visited + 1
        local role = el:attributeValue("AXRole")
        if role == "AXScrollArea" then
            local p, s = el:attributeValue("AXPosition"), el:attributeValue("AXSize")
            local sb = el:attributeValue("AXVerticalScrollBar")
            if p and s and sb and point.x >= p.x and point.x < p.x + s.w
                    and point.y >= p.y and point.y < p.y + s.h
                    and (not best or s.w * s.h < best.area) then
                local bp, bs = sb:attributeValue("AXPosition"), sb:attributeValue("AXSize")
                if bp and bs and bs.w > 0 then
                    best = {area = s.w * s.h, x = bp.x, w = bs.w}
                end
            end
        end
        if role ~= "AXWebArea" and depth < 12 then
            for _, c in ipairs(el:attributeValue("AXChildren") or {}) do
                table.insert(queue, {c, depth + 1})
            end
        end
    end
    if not best then return nil end
    return string.format("%d:%d", math.floor((best.x - frame.x) * scale),
        math.ceil((best.x + best.w - frame.x) * scale))
end

local function stitch(dir, frames, out, fixedCols)
    if frames == 1 then
        -- nothing scrolled: the single frame is the result
        local src = dir .. "/frame_0001.png"
        if not os.rename(src, out) then
            local data, f = readFile(src), io.open(out, "wb")
            if not (data and f) then error("could not write " .. out) end
            f:write(data)
            f:close()
        end
        deliver(out)
        removeDir(dir)
        return
    end
    if not hs.fs.attributes(PYTHON) then
        error("python3 not found on the login-shell PATH")
    end
    local args = {STITCH, dir, "-o", out, CONTENT_ONLY and "--content-only" or "--window",
        "--no-shift", "-e", "--prefer", "middle", "--max-overlap", "100000"}
    if fixedCols then
        table.insert(args, "--fixed-cols")
        table.insert(args, fixedCols)
    end
    local started = runTask(PYTHON, args,
        guarded(function(code, stdout, stderr)
            log.i(stdout .. stderr)
            if code ~= 0 or not hs.fs.attributes(out) then
                hs.alert.show("Scroll capture failed; frames kept in " .. dir)
                return
            end
            deliver(out)
            removeDir(dir)
        end, dir))
    if not started then error("could not start " .. PYTHON) end
end

local startCapture

function M.start()
    if running then
        stopRequested = true
        return
    end
    local ok, err = xpcall(startCapture, debug.traceback)
    if not ok then
        fail(err, session and session.dir)
        running = false
    end
end

startCapture = function()
    local win = hs.window.frontmostWindow()
    if not win then
        hs.alert.show("Scroll capture: no window")
        return
    end

    pcall(removeOldCaptures)

    -- everything that can fail before touching the pointer or keyboard
    local wid = win:id()
    local out = outputPath(win)
    local dir = os.getenv("TMPDIR") .. "scroll-capture-" .. hs.host.uuid()
    if not hs.fs.mkdir(dir) then error("could not create " .. dir) end
    local frame = win:frame()
    local step = math.floor(frame.h * STEP_FRAC)
    -- best effort: an app that doesn't report its scrollbar just gets none
    local okAx, fixedCols = pcall(scrollbarCols, win, frame.center)
    if not okAx then
        log.w("scrollbar lookup failed: " .. tostring(fixedCols))
        fixedCols = nil
    end

    session ={dir = dir, origMouse = hs.mouse.absolutePosition()}
    running, stopRequested = true, false
    hs.mouse.absolutePosition(frame.center)    -- scroll events go to the view under the pointer
    session.escKey = hs.hotkey.bind({}, "escape", function() stopRequested = true end)
    log.i("capturing window " .. wid .. ", step " .. step .. "pt, scrollbar cols " .. (fixedCols or "none"))

    local n, prev = 0, nil

    local function finish()
        log.i("finished after " .. n .. " frame(s)")
        endSession()
        if n == 0 then
            hs.alert.show("Scroll capture: no frames captured")
            removeDir(dir)
            return
        end
        stitch(dir, n, out, fixedCols)
    end

    -- Each step is sent as a trackpad gesture (began -> changed... -> ended).
    -- Mouse tweakers (Mos etc.) leave phased events alone; a plain wheel event
    -- gets reversed/smoothed by them.
    local props = hs.eventtap.event.properties
    local function scrollEvent(dy, phase)
        local e = hs.eventtap.event.newScrollEvent({0, dy}, {}, "pixel")
        e:setProperty(props.scrollWheelEventIsContinuous, 1)
        e:setProperty(props.scrollWheelEventScrollPhase, phase)
        e:post()
    end

    -- Paced like a slow drag that stops before lifting: events every TICK, then a
    -- pause, so the view sees ~zero release velocity and doesn't fling.
    local function scroll(done)
        local left, phase = step, 1                  -- kCGScrollPhaseBegan
        local tick
        tick = guarded(function()
            if left > 0 then
                local d = math.min(CHUNK, left)
                scrollEvent(-d, phase)
                phase = 2                            -- kCGScrollPhaseChanged
                left = left - d
                hs.timer.doAfter(TICK, tick)
            else
                hs.timer.doAfter(HOLD, guarded(function()
                    scrollEvent(0, 4)                -- kCGScrollPhaseEnded
                    done()
                end, dir))
            end
        end, dir)
        tick()
    end

    local function grab(path, cb)
        local started = runTask("/usr/sbin/screencapture", {"-x", "-o", "-l" .. wid, path},
            guarded(function(code) cb(code == 0 and readFile(path) or nil) end, dir))
        if not started then cb(nil) end
    end

    -- Re-capture until two grabs in a row match, so smooth-scroll animation and
    -- the rubber-band bounce at the end have finished before a frame counts.
    -- Something that never stops (a video, an animated demo) can't settle, and
    -- waiting longer doesn't give a better frame: after the first such timeout,
    -- the rest of the capture waits only long enough for the scroll itself.
    local wait = MAX_WAIT
    local function settled(path, last, deadline, cb)
        hs.timer.doAfter(POLL, guarded(function()
            grab(path, function(data)
                if not data or data == last then
                    cb(data)
                elseif hs.timer.secondsSinceEpoch() >= deadline then
                    if wait > ANIMATED_WAIT then
                        log.i("frame never settled (animated content); shorter waits from now on")
                        wait = ANIMATED_WAIT
                    end
                    cb(data)
                else
                    settled(path, data, deadline, cb)
                end
            end)
        end, dir))
    end

    local capture
    capture = function()
        if not hs.window.get(wid) then     -- window closed: stitch what we have
            log.w("window closed during capture")
            finish()
            return
        end
        local path = string.format("%s/frame_%04d.png", dir, n + 1)
        settled(path, nil, hs.timer.secondsSinceEpoch() + wait, function(data)
            if not data then
                log.e("screencapture failed for frame " .. (n + 1))
                finish()
                return
            end
            if data == prev then           -- content stopped moving: reached the end
                os.remove(path)
                finish()
                return
            end
            n, prev = n + 1, data
            log.i("frame " .. n)
            if stopRequested then
                finish()
                return
            end
            scroll(capture)
        end)
    end

    capture()
end

hs.hotkey.bind({"ctrl", "cmd", "shift"}, "2", M.start)

return M
