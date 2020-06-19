#!/bin/bash
BASEDIR="$1"
OUTDIR="$2"
tgt_lang="$3"
UNIMODAL="$4"
STYLE_0_LABEL="$5"
STYLE_1_LABEL="$6"
# In the unimodal case, POS is supposed to be the stylistic corpus

# Step 1: Create source data for tagger
for split in train test val; do
	if [ "$UNIMODAL" -eq 1 ]; then
		# UNIMODAL: The tags are deleted to create the source data for unimodal case
		cp "${BASEDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}.${STYLE_0_LABEL}" "${OUTDIR}/en${tgt_lang}_parallel.${split}.en"
		# UNIMODAL: The tags are deleted to create the source data for unimodal case
		sed -i "s/\[${STYLE_0_LABEL}[0-9]*\]//g;s/  / /g;s/^ //g;s/ $//g" ${OUTDIR}/en${tgt_lang}_parallel.${split}.en
	else
		# BIMODAL: Source data for the bimodal case is just the concatenation of the two styles
		cat ${BASEDIR}/en${tgt_lang}_parallel.${split}.en."${STYLE_0_LABEL}" ${BASEDIR}/en${tgt_lang}_parallel.${split}.en."${STYLE_1_LABEL}" > ${OUTDIR}/en${tgt_lang}_parallel.${split}.en
	fi
	# the following line performs simple strip operations on the lines
	sed -i 's/  / /g;s/^ //g;s/ $//g' ${OUTDIR}/en${tgt_lang}_parallel.${split}.en
done

# Step 2: Create target data for tagger
for split in train test val; do
	if [ "$UNIMODAL" -eq 1 ]; then
		# UNIMODAL: Target for unimodal tagger is the POS tagged data
		cp "${BASEDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}.${STYLE_0_LABEL}" "${OUTDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}"
	else
		cat ${BASEDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}."${STYLE_0_LABEL}" ${BASEDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}."${STYLE_1_LABEL}" > ${OUTDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}
	fi
done

mv ${OUTDIR}/en${tgt_lang}_parallel.val.en ${OUTDIR}/en${tgt_lang}_parallel.dev.en
mv ${OUTDIR}/en${tgt_lang}_parallel.val.${tgt_lang} ${OUTDIR}/en${tgt_lang}_parallel.dev.${tgt_lang}
