# 修复 pptxgenjs 富文本 run 产生的重复 <a:pPr>（每段只保留第一个），schema 才合法
import re, shutil, sys, zipfile

path = sys.argv[1]
tmp = path + ".tmp"

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
pPr = "{%s}pPr" % A
pTag = "{%s}p" % A

import xml.etree.ElementTree as ET
ET.register_namespace("", A)
# keep other namespaces intact on serialize
NS = re.compile(rb'<p:sld [^>]*>', re.I)

fixed = 0
with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if re.match(r"ppt/(slides|notesSlides)/[^/]+\.xml$", item.filename):
            root = ET.fromstring(data)
            for pel in root.iter(pTag):
                prs = [c for c in pel if c.tag == pPr]
                for extra in prs[1:]:
                    pel.remove(extra)
                    fixed += 1
            data = ET.tostring(root, xml_declaration=True, encoding="UTF-8", method="xml")
        zout.writestr(item, data)

shutil.move(tmp, path)
print(f"removed {fixed} redundant pPr")
