# Logistic Regression as a Neural Network

## Binary Classification

- m개의 학습 표본에 대해 for문을 돌리면서 하나씩 학습?

  - 하지만, 신경만 구현할 때는 그러지 않을 것.

    - forward pause(= forward propagation step)

    - backward pause(= backward propagation step)



- **Binary Classification**

  - (예시) "cat" 이미지
    - 1 (cat) vs 0 (non cat)
    - 64 x 64 크기의 m개의 이미지
      - n = 64 x 64 (pixel) x 3(RGB) = 12288
      - n을 x에 입력해서 label이 "cat"인지 확인

  - 단, array를 만들 때, col 1개당 데이터 1개가 들어가는게 학습시키기 더 쉬우니 주의!

- [Notation_guide.pdf](Notation_guide.pdf) 참고





## Logistic Regression

- 결과물이 0 또는 1인 이진 학습
- (예시) cat 이미지가 고양이인지 아닌지의 확률 0<= y <= 1을 알려줌
- 시그모이드(Sigmoid) 함수(0과 1 사이의 값을 가지는 S자 모양의 함수)

- 매개변수





## Logistic Regression Cost Function

- Loss(error) function: 일단 (y_hat - y)^2/2로 하겠지만 일반적으로는 비볼록하기 때문에 그렇게는 안함.
  - 보통 regression Loss function을 사용함.
- Loss function은 하나의 데이터에 대해서만, 이를 여러 데이터에 대해 합친게 Cost Function





## Gradient Descent

- Gradient Descent(기울기 강하)
  - 비용함수 J를 최소화 하는 공간매개변수 w, b를 찾는 여러 방법 중 하나.
  - 비용함수 J는 볼록한(convex) 함수.
    - 그렇기 때문에 최소값 찾기가 쉬워서 사용.
    - 초기화도 편함.
      - 무작위로 초기화 하기도 하지만,
      - 이 경우는 볼록하기 때문에 기울기 강하에 따르기 때문에 그럴 필요는 없음.







## Derivatives

- 기본적인 미분이라 생략







## More Derivative Examples

- 마찬가지로 생략







## Computation graph

- forward pass or backward pass