#!/usr/bin/env bash
# quick OpenGL / Qt diagnosis inside the runtime
Xvfb :98 -screen 0 1280x720x24 +extension GLX +render -noreset >/dev/null 2>&1 &
sleep 3
export DISPLAY=:98 LIBGL_ALWAYS_SOFTWARE=1
glxinfo -B 2>&1 | grep -E "renderer|Core profile version|Max core|direct rendering" 
echo "--- egl devices"; ls -la /dev/dri 2>&1 | head -5
echo "--- qt plugin"; QT_DEBUG_PLUGINS=1 timeout 20 ign gazebo -g 2>&1 | grep -iE "cannot load|missing|xcb|not found|Could not" | head -8
