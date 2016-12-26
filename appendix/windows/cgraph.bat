set EXT=".png"
set DOTFILE=%1
set IMGFILE=%~n1%EXT%
dot -Gdpi="96" -Tpng %DOTFILE% -o %IMGFILE%
