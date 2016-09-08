ASCIIDOC=asciidoc
ASCIIDOCTOR=asciidoctor
A2X=a2x

PDFS := $(patsubst %.asciidoc,%.pdf,$(wildcard *.asciidoc))
XMLS := $(patsubst %.asciidoc,%.xml,$(wildcard *.asciidoc))
HTMLS := $(patsubst %.asciidoc,%.html,$(wildcard *.asciidoc))
TXTS := $(patsubst %.asciidoc,%.text,$(wildcard LICENSE.asciidoc))
MANS := $(patsubst %.asciidoc,%.man,$(wildcard *_man.asciidoc))

default: html

pdf: $(PDFS)
	@echo "PDFS: $(PDFS)"

html: $(HTMLS)
	@echo "HTMLS: $(HTMLS)"

txt: $(TXTS)
  @echo "TXTS: $(TXTS)"

manf: $(MANS)
	@echo "MANS: $(MANS)"

%.text: %.asciidoc
	$(A2X) -f text $<

%.html: %.asciidoc
	$(ASCIIDOCTOR) -b html5 -a data-uri -a icons -a toc2 -o html/$@ $<
	$(ASCIIDOCTOR) -b html5 -a data-uri -a icons -o html/$@.notoc $<

%.pdf: %.asciidoc
	asciidoctor-pdf -a pdf-style=codeplay-doc-theme.yml  -b pdf -o pdf/$@ $<

%.man: %.asciidoc
	$(ASCIIDOCTOR) -b manpage -d manpage -o man/$@ $<

.PHONY: clean

clean:
	rm -Rf pdf/$(PDFS) html/$(HTMLS) man/$(MANS) $(XMLS) $(TXTS)
