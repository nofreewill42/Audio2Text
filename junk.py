
from xformers.ops import fmha

# import matplotlib.pyplot as plt; plt.imshow(fmha.BlockDiagonalGappyKeysMask.f
# rom_seqlens(q_seqlen=[2,4,3], kv_seqstarts=[0,3,13,19], kv_seqlen=[4,10,7]).materialize([6,9])); plt.show()

q_lens = [2,4,1]
kv_lens = [4,10,7]
kv_starts = [0]
for i in range(len(kv_lens)-1):
    kv_starts.append(kv_starts[-1] + kv_lens[i])
kv_starts.append(sum(kv_lens))

mask = fmha.BlockDiagonalGappyKeysMask.from_seqlens(q_seqlen=q_lens, kv_seqstarts=kv_starts, kv_seqlen=kv_lens)

import matplotlib.pyplot as plt
plt.imshow(mask.materialize([sum(q_lens), sum(kv_lens)]))
plt.show()

print(None)