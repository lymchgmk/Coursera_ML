# Introduction to Deep Learning - Andrew Ng

## What is a neural network?

- 딥러닝 : 신경망의 트레이닝, 혹은 매우 큰 신경망

  - 그렇다면 "신경망"은 무엇인가?

    - 예시) 집값을 예측하는 툴

      >집 크기에 따른 집값 예측.
      >
      >집 크기가 x, 집값이 y일 때, 이 일차함수를 만드는 신경세포.

      - ReLU(Rectified Linear Unit) 함수 : 한 동안 값이 0 이다가 직선으로 증가하는 함수?

      > 집값에 영향을 줄 수 있는 여러 요소들(방의 갯수, 가족의 크기, zip code, 이웃의 재산크기 등)을 x로 입력 받아서 집값인 y를 예측하는 것이 신경망의 역할.







## Supervised Learning with Neural Networks

- 현재 신경망과 관련하여 경제적인 가치의 대부분은 **지도학습** 을 통해서 만들어짐.

- **지도학습**에서는 x 입력값이 있으면, 해당 값과 연계되는 결과값 y를 배워야함.

  - 예시 1) 온라인 광고

    > 광고와 사용자 정보를 x 입력값으로 받음. 사람들과 유저들에게 가장 클릭할만한 광고를 보여줌.

  - 예시 2) 음성인식

    > 오디오 영상을 신경망 입력값으로 하여, 텍스트를 결과값으로 나오게 함. 예를 들면 자막.

  - 예시 3) 번역

    > 영어를 입력값, 중국어를 결과값으로.

  - 예시 4) 자율주행

    > 차 전방의 사진 + 레이더에 확인되는 정보를 기반으로, 도로에 있는 다른 차들의 위치를 파악.

- 신경망이 똑똑하게 x, y를 선정하는 것으로 상당한 가치가 있음.

  - 예를 들면, 부동산 어플같은 거는 전 세계가 공통적인 구조인데, 아마도 부동산이나 온라인 광고 산업 같은 경우, 스탠다드한 신경망을 사용하고 있을 것.

- 이미지 어플에서는 Convolutional Neural Networks(CNN) 사용.

- Sequence data, 시간적인 요소가 있는 오디오같은 데이터는 Recurrent Neural Network (RNN)을 가장 많이 사용.

  - 자율주행(Autonomous driving)은 RNN보다는 CNN과 비슷한 구조를 가지며, 레이더 정보도 있어서 다름. 그리고 커스텀 버전이나 복잡한 조합의 Hybrid neural network 구조를 사용하기도 함.



- **CNN**과 **RNN**의 차이
  - **CNN**은 이미지데이터에 많이 쓰임.
  - **RNN**은 시간적인 요소를 담고 있는 일차원적인 데이터에 주로 쓰임.



- **Structured data**와 **Unstructured data**
  - **Structured data**
    - 한마디로 데이터베이스의 데이터
    - column, row로 구성되어 의미가 명확함
  - **Unstructured data**
    - 오디오, 이미지, 텍스트 같은 데이터들
    - 내부에 들어 있는 내용을 인식해야 하는 데이터.
    - Structured data보다 컴퓨터가 인식하기 힘들어하고 사람은 인식하기 쉬워함.







## Why is Deep Learning taking off?

- Deep Learning이 요즘 뜨는 이유?
  - 원래 Deep Learning 네트워크의 개념은 수십 년 동안 존재해 왔으나,
  - 사회가 디지털화되면서, 다량의 데이터를 얻을 수 있게 됨.
    - 새로 만들어지는 데이터의 양이 증가
    - 핸드폰 안에 장착되는 IoT의 센서들(카메라, 가속도계 등)의 데이터 축적
    - 알고리즘의 혁신을 통해, 더 빠르게 계산이 가능해짐.
  - 앞으로도 GPU의 발전 등에 힘입어 더 빠르게 발전해 나갈 것.







## About this Course

- 앞으로 몇 주 동안 배울 내용의 요약

- 해당 Specialization은 5개의 코스로 이루어져 있음.

  - 1. Neural Networks and Deep Learning <- 지금 진행 중인 코스

    - Outline of this Course
      - Week 1: Introduction
      - Week 2: Basics of Neural Network programming
      - Week 3: One hidden later Neural Networks
      - Week 4: Deep Neural Networks

    1. Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
    2. Structuring your Machine Learning project
    3. Convolutional Neural Networks
    4. Natural Language Processing: Building sequence models







## Course Resources

- Question, bug reports etc.
- Contack us: feedback@deeplearning.ai
- Companies: enterprise@deeplearning.ai
- Universities: academic@deeplearning.ai