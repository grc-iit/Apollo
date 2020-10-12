#!/usr/bin/env bash

stream_name="stream_5-10_posix_memalign"
loc="$(pwd)"
install_loc="$loc/install/bin/"

echo "Downloading Stream"
curl  https://asc.llnl.gov/coral-2-benchmarks/downloads/stream_5-10_posix_memalign.c >  $stream_name.c

echo "Compiling Stream"
gcc -openmp -O3 -o $stream_name $stream_name.c

echo "Moving to install directory"
if [ ! -d "$loc/install" ]; then
  mkdir "$loc/install"
  mkdir "$install_loc"
fi
mv $stream_name $install_loc/$stream_name

echo "Making config"
echo "[STREAM]" >> probe_config.config
echo "stream_install=$install_loc" >> probe_config.config
