# python extract.py ../zeshel_full --N 4 --C 5

import argparse
import json
import os
import pathlib
import random
import shutil


def main(args):
    random.seed(args.seed)

    mens, cans = get_train_mens_cans(args.zeshel, args.C)
    print('Loaded %d training mentions and %d candidate sets (restricted to '
          'at most %d candidates each)' % (len(mens), len(cans), args.C))

    mens = [men for men in mens if men['label_document_id'] in
            cans[men['mention_id']]['tfidf_candidates'][:args.C]]
    print('Filtered %d mentions that have golds in top %d candidates' %
          (len(mens), args.C))

    random.shuffle(mens)
    mens = mens[:args.N]
    print('Extracted %d random mentions' % len(mens))

    docs, fnames, domains = get_docs(args.zeshel)
    print('Loaded %d documents across %d domains' % (len(docs), len(domains)))

    docids = {}


    for men in mens:
        # Collect context and label documents
        docids[men['context_document_id']] = True
        docids[men['label_document_id']] = True

        # Collect candidate documents
        for docid in cans[men['mention_id']]['tfidf_candidates']:
            docids[docid] = True

    fname2docids = get_fname2docids(fnames, docids)
    print('Extracted %d documents used in the mentions/candidates '
          'spanning %d domains' % (len(docids), len(fname2docids)))

    print('Writing development data...')
    create_dirs()

    for fname in fname2docids:
        with open('documents/%s' % fname, 'w') as f:
            for docid in fname2docids[fname]:
                f.write('%s\n' % json.dumps(docs[docid]))

    for name in ['train', 'heldout_train_seen', 'heldout_train_unseen', 'val',
                 'test']:
        with open('mentions/%s.json' % name, 'w') as f:
            for men in mens:
                f.write('%s\n' % json.dumps(men))

        with open('tfidf_candidates/%s.json' % name, 'w') as f:
            for men in mens:
                f.write('%s\n' % json.dumps(cans[men['mention_id']]))


def create_dirs():
    documents_path = pathlib.Path('documents')
    mentions_path = pathlib.Path('mentions')
    tfidf_candidates_path = pathlib.Path('tfidf_candidates')
    if documents_path.exists() and documents_path.is_dir():
        shutil.rmtree(documents_path)
    if mentions_path.exists() and mentions_path.is_dir():
        shutil.rmtree(mentions_path)
    if tfidf_candidates_path.exists() and tfidf_candidates_path.is_dir():
        shutil.rmtree(tfidf_candidates_path)
    documents_path.mkdir()
    mentions_path.mkdir()
    tfidf_candidates_path.mkdir()


def get_fname2docids(fnames, docids):
    fname2docids = {}
    for docid in docids:
        if not fnames[docid] in fname2docids:
            fname2docids[fnames[docid]] = {}
        fname2docids[fnames[docid]][docid] = True
    return fname2docids


def get_train_mens_cans(zeshel, C):
    mens = []
    with open(os.path.join(args.zeshel, 'mentions/train.json')) as f:
        for line in f:
            mens.append(json.loads(line))

    cans = {}
    with open(os.path.join(args.zeshel, 'tfidf_candidates/train.json')) as f:
        for line in f:  # Same order as mentions
            fields = json.loads(line)
            assert not fields['mention_id'] in cans
            cans[fields['mention_id']] = json.loads(line)
            cans[fields['mention_id']]['tfidf_candidates'] \
                = cans[fields['mention_id']]['tfidf_candidates'][:args.C]

    return mens, cans


def get_docs(zeshel):
    docs = {}
    fnames = {}
    domains = {}
    docpath = os.path.join(args.zeshel, 'documents')
    for fname in os.listdir(docpath):
        with open(os.path.join(docpath, fname)) as f:
            for line in f:
                fields = json.loads(line)
                docs[fields["document_id"]] = fields
                fnames[fields["document_id"]] = fname
                domains[fname] = True
    return docs, fnames, domains


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('zeshel', type=str, default='../zeshel_full')
                        help='path to zeshel directory containing mention/, '
                        'documents/, and tfidf_candidates/')
    parser.add_argument('--N', type=int, default=4,
                        help='number of training mentions to extract '
                        '[%(default)d]')
    parser.add_argument('--C', type=int, default=5,
                        help='gold label must be in top-C in candidates'
                        '[%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')

    args = parser.parse_args()
    main(args)
