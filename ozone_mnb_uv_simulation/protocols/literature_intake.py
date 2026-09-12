import os, json, hashlib, shutil, zipfile

SRC = '/mnt/c/Users/pc/Desktop/DFT学习/DFT文献'
DST = '/mnt/e/dft-service/ozone_mnb_uv_simulation/references/literature'
os.makedirs(DST, exist_ok=True)

def sha(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()

manifest = []
seen = {}
for name in sorted(os.listdir(SRC)):
    sp = os.path.join(SRC, name)
    if not os.path.isfile(sp):
        continue
    ext = os.path.splitext(name)[1].lower()
    h = sha(sp)
    fmt = {'.pdf': 'PDF', '.zip': 'ZIP', '.html': 'HTML', '.htm': 'HTML'}.get(ext, ext or 'unknown')
    entry = dict(source_filename=name, sha256=h[:16], sha256_full=h,
                 size_bytes=os.path.getsize(sp), format=fmt,
                 copied_to='references/literature/' + name,
                 duplicate_of=None, version_note=None)
    if h[:16] in seen:
        entry['duplicate_of'] = seen[h[:16]]['source_filename']
        entry['duplicate_type'] = 'exact (same SHA-256)'
    seen[h[:16]] = entry
    # near-duplicate by base name
    base = name.replace(' (1)', '').replace(' (2)', '')
    if base != name:
        entry['version_note'] = 'numbered copy of "%s" — check same-hash or version pair' % base
    # copy (do not overwrite a different file with same name; verify by hash)
    dp = os.path.join(DST, name)
    if os.path.exists(dp):
        dh = sha(dp)
        if dh == h:
            entry['copy_action'] = 'already present (hash-verified reuse)'
        else:
            stem, e2 = os.path.splitext(name)
            dp = os.path.join(DST, stem + '__src2' + e2)
            shutil.copy2(sp, dp)
            entry['copy_action'] = 'name collision with different content -> copied as ' + os.path.basename(dp)
    else:
        shutil.copy2(sp, dp)
        entry['copy_action'] = 'copied'
    # first-page text for PDFs (skip huge files for speed except key ones)
    if fmt == 'PDF':
        try:
            from pypdf import PdfReader
            r = PdfReader(sp)
            entry['n_pages'] = len(r.pages)
            txt = (r.pages[0].extract_text() or '')[:1200]
            entry['first_page_text_head'] = txt.replace('\n', ' ')[:600]
        except Exception as e:
            entry['n_pages'] = None
            entry['first_page_text_head'] = 'EXTRACT_ERROR: %r' % e
    manifest.append(entry)

# duplicate/version mapping by base name
base_map = {}
for e in manifest:
    b = e['source_filename'].replace(' (1)', '').replace(' (2)', '')
    base_map.setdefault(b, []).append(e['source_filename'])
versions = {b: v for b, v in base_map.items() if len(v) > 1}

# ZIP member path check (do NOT extract yet)
zip_report = []
for e in manifest:
    if e['format'] != 'ZIP':
        continue
    zp = os.path.join(SRC, e['source_filename'])
    try:
        with zipfile.ZipFile(zp) as z:
            members = z.namelist()
            unsafe = [m for m in members if m.startswith('/') or '..' in m or ':' in m]
            zip_report.append(dict(zip_file=e['source_filename'],
                                   n_members=len(members),
                                   members=members[:30],
                                   unsafe_members=unsafe,
                                   path_check='FAIL' if unsafe else 'PASS'))
    except Exception as ex:
        zip_report.append(dict(zip_file=e['source_filename'], error=repr(ex)))

out = dict(job='JOB-2026-0906-023 intake',
           source_dir=SRC,
           n_files=len(manifest),
           manifest=manifest,
           version_groups=versions,
           zip_report=zip_report,
           note='desktop originals untouched (copy-only); exact duplicates identified by SHA-256')
json.dump(out, open('/mnt/e/dft-service/ozone_mnb_uv_simulation/references/literature/manifest.json', 'w'),
          indent=2, ensure_ascii=False)
print('files:', len(manifest))
print('version groups:')
for b, v in versions.items():
    print('  ', b, '<-', v)
print('zip report:', json.dumps(zip_report, ensure_ascii=False)[:800])
dup = [e for e in manifest if e['duplicate_of']]
print('exact duplicates:', [(d['source_filename'], '->', d['duplicate_of']) for d in dup])
