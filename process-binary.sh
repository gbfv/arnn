#!/bin/bash
#
if [ "$#" -ne 1 ]; then
  echo "Usage: process-binary.sh <filename>"
  exit 1
fi

echo "Extracting..."
rizin -e asm.functions=false -e asm.lines=false -e asm.comments=false -e hex.cols=1 -c "aaaa;b 1;pdf @@F > addr_debug.log; pxf @@F > bytes_debug.log" -qq d3d10core.dll
echo "Processing files..."
cat addr_debug.log | cut -w -f 1 >addr.log
cat bytes_debug.log | cut -w -f 1-2 --output-delimiter=" " >bytes.log
