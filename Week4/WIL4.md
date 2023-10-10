Object Detection
input: RGB image
output: label(어떤 클래스), bbox info(그 클래스가 어디에 위치해있는지, x, y, width, height)

image classification(하나의 이미지. 하나의 라벨) vs object detection(더 실용적)
object detection output: 위치+라벨  4개의 객체 -> output 4개
따라서 고해상도 image가 들어가야함. 학습시간 up

image(input) -> CNN -> Vector1 -> FC Layer -> Vector2(class1000개) -> softmax loss
손실함수 2개임 (softmax / L2 loss) -> 근데 손실함수는 하나여야해서 이 두 개를 더함.(weight sum loss)

하나의 객체에 대한 위치 예측. 그리고 실제 위치. 그리고 비교 (loss)
근데 문제 발생 -> 이미지는 단일 객체가 아님

sliding window -> 위 문제를 해결할 수 있는 방법
작은 박스(윈도우) 안에 뭐가 있는지 분석. 그리고 그 박스를 몇번 옮긴다.
옮기는 횟수(N) = (W-(w'-1))x(H-(h'-1))
그리고 window의 크기는 다양함. -> 모든 윈도우 개수 시그마(h=1~H)시그마(w=1~W)(W-w+1)(H-h+1) = H(H+1)/2 * W(W+1)/2

문제1. 이미지 크기가 크다면 -> 많은 횟수의 image classification 해야함.(시간 소요 up, 필요한 컴퓨팅 파워 up) => 실행불가
문제2. 동일한 객체를 계속해서 무분별하게 식별 => (진행하더라도) 정확하지 않은 결과

영역제안(region proposal) : 어떻게 하면 객체를 포함할 확률이 높은 이미지 영역을 찾을 수 있을까?
selective search(나중에 신경망으로 교체됨)

R-CNN (selective search알고리즘을 사용한 object detection)
1) selective search로 ROI 뽑기 2) region size process(warping) 3) compute CNN features(피쳐값 뽑는다) 4) linear SVM로 classify
Bbox reg / SVMs 위치 정보 예측

IOU(intersection over union) = (Area of overlap. 교집합) / (area of union. 합집합)
IOU가 1에 가깝다 => 같은 object를 가르키고 있을 확률 높음 (>0.5: 보통. >0.7: 꽤 좋음. >0.9: 거의 완벽)
threshold는 user가 정함.

Non-Max Suppression (NMS)
가장 높은 score 가진 박스 선택 -> 나머지 score를 가진 박스들과 IOU 비교해서 threshold(예. 0.7) 이상의 IOU를 가진 박스들 제거 -> 반복
단점: 유의미한 boundy box(bbox)를 제거할 수 있다. (여러 객체들이 겹치는 이미지에서)

Mean Average Precision(mAP)
각 카테고리별 Average Precision을 계산
threshold설정에 따라 정밀도& 재현율값이 변화. 1. threshold high -> bbox 수 low(신중해짐) / 2. threshold low -> bbox 난사(무분별)
Precision: 모델이 positive라고 예측한 것 중에서 실제 True인 비율
Recall: 실제 맞게 예측한 것 중에서 모델이 Positive라고 예측한 비율
precision x recall curve -> 아래의 넓이=> Average Precision

Fast R-CNN -> R-CNN의 단점을 보완(너무 오래걸림)
- CNN과 Resize의 순서를 바꾸어 속도 개선 - Classification으로 SVM에서 linear classification 사용
ROI pooling => resize한다.

Faster R-CNN / Region Proposal Network(RPN) -> Faster R-CNN도 오래걸림 이 단점 해결
1. CNN을 통과해 feature map 뽑기 2. region proposal network로 region proposal 3. roi pooling 거쳐 classification과 bbox regression
- 앵커박스(윈도우 같은거) 안에 객체를 포함하고 있는지 아닌지를 판단 - 1x1 binary classification 수행 -> classification loss (객체가 있냐 없냐)
- 객체를 포함하고 있다면 구체적으로 중심을 찾아 이동하고 resize를 해주는 역할 (bbox regression loss)
