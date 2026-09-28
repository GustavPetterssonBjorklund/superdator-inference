{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    pkgs.python311
    pkgs.python311Packages.venvShellHook

    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libGL
    pkgs.glib

    pkgs.xorg.libX11
    pkgs.xorg.libXext
    pkgs.xorg.libXrender
    pkgs.xorg.libXtst
    pkgs.xorg.libXi
    pkgs.xorg.libSM
    pkgs.xorg.libICE
    pkgs.xorg.libxcb
    pkgs.xorg.xcbutil
    pkgs.xorg.xcbutilimage
    pkgs.xorg.xcbutilkeysyms
    pkgs.xorg.xcbutilrenderutil
    pkgs.xorg.xcbutilwm
  ];

  venvDir = ".venv";

  postVenvCreation = ''
    pip install --upgrade pip
  '';

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.libGL
      pkgs.glib

      pkgs.xorg.libX11
      pkgs.xorg.libXext
      pkgs.xorg.libXrender
      pkgs.xorg.libXtst
      pkgs.xorg.libXi
      pkgs.xorg.libSM
      pkgs.xorg.libICE
      pkgs.xorg.libxcb
      pkgs.xorg.xcbutil
      pkgs.xorg.xcbutilimage
      pkgs.xorg.xcbutilkeysyms
      pkgs.xorg.xcbutilrenderutil
      pkgs.xorg.xcbutilwm
    ]}:$LD_LIBRARY_PATH

    source .venv/bin/activate

    export NIX_SHELL_NAME="Inference Shell"
    if [ -n "$BASH_VERSION" ]; then
      export PS1="\[\e[1;35m\][nix:$NIX_SHELL_NAME]\[\e[0m\] \[\e[1;36m\]\w\[\e[0m\] \$ "
    elif [ -n "$ZSH_VERSION" ]; then
      export PROMPT="%F{magenta}[nix:$NIX_SHELL_NAME]%f %F{cyan}%~%f %# "
    fi
  '';
}
