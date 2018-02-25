ifeq ($(OS),Windows_NT)
	SHELL=%WINDIR%/cmd.exe
	DROPBOX_PUBLIC=$(USERPROFILE)/Dropbox/Public
else
	SHELL=/usr/bin/env sh
	DROPBOX_PUBLIC=$(HOME)/Dropbox/Public
endif
PDFLATEX = pdflatex
REPORT = report
TITLE = title
TASK = task
BIBTEX = bibtex
RM = rm -f
GREP = grep
SUBDIR_ROOTS := sections
DIRS := . $(shell find $(SUBDIR_ROOTS) -type d)
GARBAGE_PATTERNS := *.aux *.log *.out *.toc *.gz *.gz\(busy\) *.blg *.bbl
GARBAGE := $(foreach DIR,$(DIRS),$(addprefix $(DIR)/,$(GARBAGE_PATTERNS)))

diplom_only: $(REPORT).pdf

all: $(REPORT).pdf $(TASK).pdf $(TITLE).pdf


fast: *.tex
	latexmk -pdf -pdflatex="pdflatex" $(REPORT)
	mv $(REPORT).pdf $(REPORT)-`date +'%m-%d-%H-%M-%S'`.pdf

fastcheck: *.tex
	$(PDFLATEX) $(REPORT)
	$(BIBTEX) $(REPORT).aux
	$(PDFLATEX) $(REPORT)
	$(PDFLATEX) $(REPORT)
	mv $(REPORT).pdf $(REPORT)-`date +'%m-%d-%H-%M-%S'`.pdf


$(REPORT).pdf: *.tex
	$(PDFLATEX) $(REPORT)
	$(BIBTEX) $(REPORT).aux
	$(PDFLATEX) $(REPORT)
	$(PDFLATEX) $(REPORT)


$(TASK).pdf: *.tex
	$(PDFLATEX) $(TASK)


$(TITLE).pdf: *.tex
	$(PDFLATEX) $(TITLE)


cleanall: clean
	$(RM)  *.pdf

.PHONY: clean
clean:
	rm -rf $(GARBAGE)
