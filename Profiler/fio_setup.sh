#!/usr/bin/env bash
loc="$(pwd)"
install_loc="$loc/install/"
zip_name="fio-master" # this is the name the archive extracts to.
unzip_loc="$loc/$zip_name/" ## need to figure this -d puts it in that folder

## get and make fio
echo "Downloading FIO"
wget -O $zip_name.zip https://github.com/axboe/fio/archive/master.zip
unzip -o $zip_name.zip
cd "$zip_name"
echo "Compiling and installing"
bash configure --prefix="$install_loc"
make
make install
cd -

## make config file
echo "Making config file"
echo "[FIO]" >> probe_config.config
echo "fio_install=$install_loc/bin/" >> probe_config.config
