#!/bin/bash
set -u
BASEDIR="$1"
OUTDIR="$2"
tgt_lang="$3"
tgt_lang_tag="$4"
UNIMODAL="$5"
POS="$6"
NEG="$7"

for split in train test val; do
	if [ "$UNIMODAL" -eq 1 ]; then
		cp ${BASEDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}."$POS"  ${OUTDIR}/en${tgt_lang_tag}_parallel.${split}.en
	else
		cat ${BASEDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}."$POS" ${BASEDIR}/en${tgt_lang}_parallel.${split}.${tgt_lang}."$NEG" > ${OUTDIR}/en${tgt_lang_tag}_parallel.${split}.en
	fi
done

for split in train test val; do
	if [ "$UNIMODAL" -eq 1 ]; then 
		cp ${BASEDIR}/en${tgt_lang}_parallel.${split}.en."$POS" ${OUTDIR}/en${tgt_lang_tag}_parallel.${split}.${tgt_lang_tag}
		sed -i "s/\[${NEG}[0-9]*\]//g;s/  / /g;s/^ //g;s/ $//g" ${OUTDIR}/en${tgt_lang_tag}_parallel.${split}.${tgt_lang_tag}
	else
		cat ${BASEDIR}/en${tgt_lang}_parallel.${split}.en."$POS" ${BASEDIR}/en${tgt_lang}_parallel.${split}.en."$NEG" > ${OUTDIR}/en${tgt_lang_tag}_parallel.${split}.${tgt_lang_tag}
	fi
done

mv ${OUTDIR}/en${tgt_lang_tag}_parallel.val.en ${OUTDIR}/en${tgt_lang_tag}_parallel.dev.en
mv ${OUTDIR}/en${tgt_lang_tag}_parallel.val.${tgt_lang_tag} ${OUTDIR}/en${tgt_lang_tag}_parallel.dev.${tgt_lang_tag}

