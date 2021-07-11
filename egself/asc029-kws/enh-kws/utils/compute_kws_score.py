import sys

if len(sys.argv) != 5:
  print("python {} <hyp> <token_list> <ref> <outdir>".format(sys.argv[0], file=sys.stderr))
  exit(1)
else:
  hyp = sys.argv[1]
  token_list = sys.argv[2]
  ref = sys.argv[3]
  outdir = sys.argv[4]

def LoadHyp(hyp,vocab):
  hyp_ = {}
  for x in open(hyp):
    uttid,idx = x.strip().split(' ',1)
    hyp_[uttid] = vocab[idx]

  return hyp_

def LoadTokenlist(token_list):
  vocab = {}
  idx = 0
  for token in open(token_list):
    if token not in vocab.values():
      vocab[str(idx)] = token.strip()
      idx += 1
    else:
      print("token is not unique",file=sys.stderr)
      exit(1)

  return vocab

def LoadRef(ref):
  ref_ = {}
  for x in open(ref):
    uttid,lab = x.strip().split(' ',1)
    ref_[uttid] = lab
  return ref_

def ComputeScore(hyp,ref):
  assert len(hyp) == len(ref)
  
  total_num = len(hyp)
  keyword_num = 0
  nonkeyword_num = 0
  far_num = 0
  frr_num = 0
  accuracy_num = 0
  for uttid,lab in hyp.items():
    if lab == ref[uttid]:
      accuracy_num += 1
    elif lab == 'FREETEXT':
      frr_num += 1
    elif lab == 'WUKONG_WUKONG':
      far_num += 1

    if ref[uttid] == 'FREETEXT':
      nonkeyword_num += 1
    if ref[uttid] == 'WUKONG_WUKONG':
      keyword_num += 1

  print("the number of total utts: {}".format(total_num))
  print("the number of keyword utts: {}".format(keyword_num))
  print("the number of nonkeyword utts: {}".format(nonkeyword_num))

  acc = accuracy_num / total_num
  far = far_num / nonkeyword_num
  frr = frr_num / keyword_num

  print("ACC: {}".format(acc))
  print("FAR: {}".format(far))
  print("FRR: {}".format(frr))


def WriteResults():
  pass


def main():
  vocab = LoadTokenlist(token_list)
  hyp_ = LoadHyp(hyp,vocab)
  ref_ = LoadRef(ref)
  ComputeScore(hyp_,ref_)


if __name__ == '__main__':
  main()
