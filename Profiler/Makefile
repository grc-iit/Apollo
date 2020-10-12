
all:
	touch probe_config.config
	rm probe_config.config
	bash fio_setup.sh
	bash stream_setup.sh
	bash install_requirements.sh

clean:
	rm -rf fio-master*
	rm -f stream_5-10_posix_memalign.c

uninstall :
	rm -rf install
	touch probe_config.config
	rm probe_config.config
