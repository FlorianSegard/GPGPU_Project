0. If you're using Nix on the OpenStack, use the provided flake.

```
nix develop
```

1. Build the project (in Debug or Release) with cmake

```
export buildir=... # pas dans l'AFS
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Debug
```

or

```
cmake -S . -B $buildir -DCMAKE_BUILD_TYPE=Release
```

2.
Run with

```
$buildir/stream --mode=[gpu,cpu] <video.mp4> [--output=output.mp4]
```

3.
Edit your cuda/cpp code in */Compute.*
