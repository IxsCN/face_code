

# Face Expression Recognition Strong Baseline
## content

## train

```python
cd tools

export PYTHONPATH=..

python train.py
```


1. batchsize 600
(gaussian 600/4*3 = 450)
(ran model 72  0.87 -> 0.78)

2. optimizer 
adam  (scn + ce)
sgd   (scn + wloss)
rmsprop (make no sense)

3. learning_rate
0.001 (ce, note! val in log file is wrong)
0.001 (finetune wloss)

4. momentum 
0.9 (adam)
0.0 (sgd)

5. weight_decay 
0.9 (adam & sgd)

6. drop_rate
0 (have not use, wait to check if it is work)

7. model
baseline(acc < 0.4, maybe bug)
scn(acc = 0.87)  (you can rename)
ran(acc = 0.78)

8. relabel_epoch
have not use

9. ranking
have not use

10. alpha(wloss)
0.0125

11. update w
not good

12. w val
have not change

13. pretrained model
(nouse, ...)
(use, acc = 0.87)
(use ourselves, todo)


14. freeze network
wloss work (idea, backbone lr = 1e-5,
todo. study
)
```
# optimizer = torch.optim.Adam([
#         {'params': model.features.parameters(), 'lr':1e-5},
#         # {'params': model.fc.parameters(), 'lr': 1e-3},
#         # {'params': model.alpha.parameters(), 'lr': 1e-5}
#     ], weight_decay=cfg.TRAIN.weight_decay)
```

15. imbalanced sample
raf db (ran, acc = 0.78)


15+. wloss -> imbalance loss


16. landmark crop


17. landmark gaussian


18. finetune wloss on ce model
sgd(train wloss not on ce acc=0.84, epoch = 300)
adam(train wloss not on ce acc=0.86, epoch = 30)

sgd(train wloss finetune on ce)



















```
这个是其中一个同学共享给我的， 请你查看，&nbsp; 谢谢各位的分享！！
…

------------------&nbsp;原始邮件&nbsp;------------------
发件人:                                                                                                                        "kaiwang960112/Self-Cure-Network"                                                                                    <notifications@github.com&gt;;
发送时间:&nbsp;2020年10月27日(星期二) 中午11:16
收件人:&nbsp;"kaiwang960112/Self-Cure-Network"<Self-Cure-Network@noreply.github.com&gt;;
抄送:&nbsp;"杨林"<185284033@qq.com&gt;;"Comment"<comment@noreply.github.com&gt;;
主题:&nbsp;Re: [kaiwang960112/Self-Cure-Network] 您好，请问可以分享SCN的预训练模型吗？(*^▽^*) (#24)






非常感谢，之前因为一些原因没有复现出来，所以才向您请求预训练模型。之后问题解决已经复现出来了。非常感谢您！

您好，方便发一下预训练模型吗？谢谢！1548360044@qq.com

—
You are receiving this because you commented.
Reply to this email directly, view it on GitHub, or unsubscribe.


从QQ邮箱发来的超大附件

epoch70_acc0.8719.pth (128.03M, 2020年11月27日 21:59 到期)进入下载页面：http://mail.qq.com/cgi-bin/ftnExs_download?t=exs_ftn_download&k=7d363834fce09c93f603333f4265514b02575a06505156061a570901064857540e07150c5c53524900575c0454045355005009016470630147595b5c53553c055455081a5c52525d19464c5c6458&code=7684decd

```





### baseline




### self cure network




## Install

```
git clone https://github.com/stoneyang159/face_expression_recognization_strong_baseline
cd fer_strong_baseline && python setup.py install
```

### Baseline Performance



###  Tools
#### Face detect by MTCNN
See `examples/face_detect_mtcnn.py`
```
import cv2
import os
import torch
from fer_pytorch.face_detect import  MTCNN

mtcnn = MTCNN(
        image_size = 224,
        min_face_size = 40,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
bboxs, scores, landmarks = mtcnn.detect(img, landmarks=True)
for box, score, points in zip(bboxs,scores,landmarks):
        box[2] = box[2] - box[0] # w
        box[3] = box[3] - box[1] # h
        cv2.rectangle(img, tuple([int(v) for v in box.tolist()]), (255,0,0),3,16)
        for p in points:
            cv2.circle(img, tuple([int(v) for v in p.tolist()]),5, (0,255,0),3,16)
img_path = img_path[:-4]+'_det.jpg'
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(img_path, img)
```

