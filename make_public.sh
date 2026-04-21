#!/bin/bash

OUT_DIR="../8be030-mia/"
DOCS_DIR="../8be030-mia-docs/"

# 1. create public version with all removed answers

# check if the directory exists
if [ -d $OUT_DIR ]
then
    echo "$OUT_DIR directory already exists!"
else 
    mkdir $OUT_DIR && echo "Creating directory $OUT_DIR..."
fi

# if the directory exists, sync the two directories
if [ -d $OUT_DIR ]
then
	if [ "$(ls -A $OUT_DIR)" ]; then
        echo "$OUT_DIR directory is not empty, copying new files, adding updates..."
        rsync -raz --update --delete ./* $OUT_DIR
        rm -fr $OUT_DIR/documentation/
        rm -f $OUT_DIR/make_public.sh
        rm -f $OUT_DIR/software.md

        # remove all lines between the lines #!studentstart and #!studentend
        find $OUT_DIR/code/ -type f -name '*.py' -exec sed -i -e '/#!studentstart/,/#!studentend/d;' {} \;

	else
        echo "$OUT_DIR is empty, copying all files.."
        cp -r * $OUT_DIR
        rm -fr $OUT_DIR/documentation/
        rm -f $OUT_DIR/make_public.sh
        rm -f $OUT_DIR/software.md

        # remove all lines between the lines #!studentstart and #!studentend
        find $OUT_DIR/code/ -type f -name '*.py' -exec sed -i -e '/#!studentstart/,/#!studentend/d;' {} \;

	fi
else
	echo "Directory $OUT_DIR not found."
fi


# 2. create another folder for online documentation

# check if the directory exists, overwrite files in it (no deletion)

if [ -d $DOCS_DIR ]
then
    echo "$DOCS_DIR directory already exists!"
    echo "Copying new files..."
    cp -rf ./documentation/* $DOCS_DIR
    cp -rf $OUT_DIR/data/ $DOCS_DIR/docs/source/
    cp -rf $OUT_DIR/reader/ $DOCS_DIR/docs/source/
    cp -rf $OUT_DIR/code/ $DOCS_DIR/docs/source/
    cp -f $OUT_DIR/README.md $DOCS_DIR/docs/source/
else 
    mkdir $DOCS_DIR && echo "Creating directory $DOCS_DIR..."
    echo "Copying new files..."
    cp -rf ./documentation/* $DOCS_DIR
    cp -rf $OUT_DIR/data/ $DOCS_DIR/docs/source/
    cp -rf $OUT_DIR/reader/ $DOCS_DIR/docs/source/
    cp -rf $OUT_DIR/code/ $DOCS_DIR/docs/source/
    cp -f $OUT_DIR/README.md $DOCS_DIR/docs/source/
fi

echo "Done."
read 