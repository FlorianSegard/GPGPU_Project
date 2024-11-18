# GPGPU: Motion Estimation


0. If you're using Nix on the OpenStack, use the provided flake.

```
nix develop
```

1. Compile the project (in Debug or Release) from project root with cmake

```
./full_recompile_and_run.sh Debug
```

**or**

```
./full_recompile_and_run.sh 
# or
./full_recompile_and_run.sh Release
```

[OPTIONAL] Re-compile the project with make (after small changes inside the source code)

```
./fast_compile_and_run.sh
```

2.
Run with

```
$buildir/stream --mode=[gpu,cpu] <video.mp4> [--output=output.mp4] \
[--opening-size=3] [--th-low=3] [--th-high=30] [--number-frame=100]
```

3.
Edit your cuda/cpp code in */Compute.*
