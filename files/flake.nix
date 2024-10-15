{
  description = "GPGPU course environment";

  inputs = {
    nixpie.url = "git+https://gitlab.cri.epita.fr/forge/infra/nixpie";
  };

  outputs =
    { self, nixpie }:
    let
      system = "x86_64-linux";
      inherit (nixpie.inputs.nixpkgs) lib;
      inherit (lib) attrValues;
      pkgs = import nixpie.inputs.nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
        overlays = (attrValues nixpie.overlays) ++ [ nixpie.overrides.${system} ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "cuda-env-shell";
        buildInputs = with pkgs; [
          git
          gitRepo
          gnupg
          autoconf
          curl
          procps
          gnumake
          utillinux
          m4
          gperf
          unzip
          cmake
          linuxPackages.nvidia_x11
          gcc11
          coreutils
          libGLU
          libGL
          xorg.libXi
          xorg.libXmu
          freeglut
          xorg.libXext
          xorg.libX11
          xorg.libXv
          xorg.libXrandr
          zlib
          pngpp
          tbb
          gbenchmark
          ncurses5
          stdenv.cc
          binutils
          cudaPackages.cuda_nvprof
          gst_all_1.gstreamer
          gst_all_1.gst-plugins-base
          gst_all_1.gst-plugins-ugly
          gst_all_1.gst-plugins-bad
          gst_all_1.gst-plugins-good
        ];
        shellHook = with pkgs; ''
          export LD_LIBRARY_PATH=${cudaPackages.cuda_nvprof.lib}/lib:$LD_LIBRARY_PATH
          export CUDAHOSTCXX=${pkgs.gcc11}/bin/g++
          export NVCC_PREPEND_FLAGS="-ccbin ${pkgs.gcc11}/bin/g++"
        '';
      };
    };
}
