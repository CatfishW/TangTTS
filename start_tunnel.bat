@echo off
setlocal EnableDelayedExpansion

set "REMOTE_USER=lobin"
set "REMOTE_HOST=vpn.agaii.org"
set "REMOTE_PORT=8888"
set "LOCAL_PORT=8888"
set "RECONNECT_DELAY=5"

echo === LLM Server Reverse SSH Tunnel ===
echo Forwarding: %REMOTE_HOST%:%REMOTE_PORT% -^> 127.0.0.1:%LOCAL_PORT%

REM Check local server
curl -s --max-time 2 http://127.0.0.1:%LOCAL_PORT%/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Local server not responding on port %LOCAL_PORT%
)

set ATTEMPT=0

:LOOP
set /a ATTEMPT+=1
echo [%TIME:~0,8%] Starting tunnel (attempt #!ATTEMPT!)...

ssh -R %REMOTE_PORT%:127.0.0.1:%LOCAL_PORT% ^
    -o ServerAliveInterval=30 ^
    -o ServerAliveCountMax=3 ^
    -o ExitOnForwardFailure=yes ^
    -o StrictHostKeyChecking=no ^
    -N %REMOTE_USER%@%REMOTE_HOST%

REM If ssh exits with 0 (success), break loop
if %ERRORLEVEL% EQU 0 goto :END

echo [WARN] Disconnected. Reconnecting in %RECONNECT_DELAY%s...
timeout /t %RECONNECT_DELAY% /nobreak >nul
goto :LOOP

:END
echo Tunnel stopped normally.
endlocal
