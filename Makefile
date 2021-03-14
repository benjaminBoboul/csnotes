
CC = lualatex
DEPENDENCIES := $(shell find notes/machines-learning/* -name *.tex)

.PHONY=machine-learning

machine-learning: ${DEPENDENCIES}
	$(CC) --output-directory=. --shell-escape notes/$@/$@.tex
	makeglossaries -d . $@
	biber -output-directory=. $@
	$(CC) --output-directory=. --shell-escape notes/$@/$@.tex

clear:
	find ./* -name *.{aux,log,pdf} -delete
