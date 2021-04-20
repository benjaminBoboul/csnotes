TEX = lualatex
TEXFLAGS = --shell-escape
.PHONY = default clean
TEXFILES = find . -name "*.tex"



default: out/cs-notes.pdf

out:
	mkdir -p $@

out/%.pdf: out notes/**/*.tex
	$(TEX) ${TEXFLAGS} --output-directory=$(@D) $*.tex
	makeglossaries -d $(@D) $*
	biber --output-directory $(@D) $*
	$(TEX) ${TEXFLAGS} --output-directory=$(@D) $*.tex