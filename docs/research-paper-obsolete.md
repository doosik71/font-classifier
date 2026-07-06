# 동적 후보 선택 기반 Top-k Relaxed Negative Learning을 이용한 시각적 유사성 인지 폰트 인식

## Visual Similarity-Aware Font Recognition via Dynamic Top-k Relaxed Negative Learning

## 초록

본 논문은 렌더링된 글자 이미지로부터 사용된 폰트 클래스를 식별하는 폰트 인식 문제를 다룬다. 기존 단일 라벨 기반 학습은 정답 클래스를 제외한 모든 클래스를 동일한 음성 클래스로 취급하지만, 실제 폰트 데이터에서는 서로 다른 폰트 이름을 갖더라도 시각적으로 매우 유사한 클래스들이 존재할 수 있다. 이러한 경우 기존 학습 방식은 시각적으로 유사한 비정답 클래스까지 강하게 억제하여 과도하게 경직된 지도 신호를 제공한다.

이를 완화하기 위해 본 논문은 동적 후보 선택 기반 Top-$k$ Relaxed Negative Learning을 제안한다. 제안 방법은 사전 학습된 인코더나 외부 유사도 행렬 없이, 모델의 현재 sigmoid activation을 이용해 정답 클래스를 제외한 상위 $k$개의 비정답 클래스를 ambiguous candidate로 선택한다. 이 후보들은 정답으로 강제되지 않고 negative loss에서 제외되는 완화 대상으로 처리되어, 모델이 정답 클래스의 식별력을 유지하면서도 시각적으로 유사한 클래스에 대한 multi-hot activation을 허용하도록 한다.

제안 손실은 정답 클래스에 대한 positive loss, ambiguous candidate를 제외한 relaxed negative loss, 과도한 activation 확산을 방지하는 sparsity regularization으로 구성된다. 또한 학습 초기의 불안정성을 줄이기 위해 warm-up 이후 Top-$k$ Relaxed Negative Loss의 비중을 점진적으로 증가시키는 curriculum 전략을 적용한다. 이를 통해 본 방법은 폰트 이름 기반 라벨 체계를 유지하면서도 클래스 간 시각적 유사성과 불확실성을 학습 과정에 반영할 수 있다.

## 1. 서론

본 연구는 렌더링된 글자 이미지로부터 해당 이미지에 사용된 폰트 클래스를 식별하는 폰트 인식 문제를 다룬다. 전체 폰트 집합은 약 3,000여 종의 폰트로 구성되어 있으며, 각 폰트는 데이터셋에서 고유한 폰트 이름을 기준으로 하나의 클래스로 정의된다. 따라서 학습 데이터의 각 샘플은 특정 폰트 이름에 대응되는 단일 클래스 라벨을 갖는다.

일반적인 폰트 인식 설정에서는 서로 다른 폰트 이름을 갖는 클래스들이 서로 배타적인 범주라고 가정한다. 이 가정하에서 모델은 입력 이미지가 주어졌을 때 3,000여 개의 클래스 중 하나를 정답 클래스로 예측하도록 학습된다. 즉, 각 샘플의 정답 라벨은 하나의 클래스에만 할당되며, 나머지 모든 클래스는 음성 클래스로 간주된다.

그러나 실제 폰트 데이터에서는 클래스 이름이 다르더라도 렌더링된 이미지 상에서 매우 유사한 시각적 형태를 보이는 폰트들이 존재할 수 있다. 이러한 유사성은 획의 두께, 자폭, 곡률, 세리프의 형태, 글자 간 균형, 전체적인 인상 등 다양한 조형적 요소에서 나타날 수 있다. 특히 제한된 글자 집합, 낮은 해상도, 작은 이미지 크기, 또는 특정 문자에 국한된 관측 조건에서는 서로 다른 폰트 클래스 간의 시각적 차이가 더욱 미세해질 수 있다.

이로 인해 폰트 인식 문제는 단순한 다중 클래스 분류 문제보다 더 복잡한 특성을 갖는다. 서로 다른 클래스가 시각적으로 충분히 구분 가능하다는 전제가 항상 성립하지 않기 때문이다. 특정 입력 이미지가 주어졌을 때, 해당 이미지의 라벨은 하나의 폰트 이름으로 지정되어 있지만, 시각적 관점에서는 다른 일부 폰트 클래스와 높은 유사성을 가질 수 있다. 그럼에도 불구하고 기존의 단일 클래스 타깃 기반 학습은 정답 클래스 이외의 모든 클래스를 동일하게 오답으로 취급한다.

이러한 학습 방식은 시각적으로 유사한 폰트 클래스가 존재하는 상황에서 모델 학습에 불리하게 작용할 수 있다. 모델이 정답 클래스와 시각적으로 가까운 다른 클래스를 높은 확률로 예측하더라도, 기존 학습 목표에서는 이를 완전한 오분류로 간주한다. 결과적으로 모델은 실제 시각적 구조를 반영하기보다 클래스 이름 간의 엄격한 배타성을 과도하게 학습하게 되며, 이는 세밀한 폰트 인식 성능 저하로 이어질 수 있다.

따라서 본 연구에서 다루는 핵심 문제는 다음과 같이 정의된다. 약 3,000여 개의 폰트 이름 기반 클래스가 주어진 상황에서, 클래스 간 시각적 유사성이 존재함에도 불구하고 각 샘플을 단일 클래스 타깃으로만 학습할 경우 발생하는 지도 신호의 경직성과 그로 인한 인식 성능 저하 문제를 분석하고자 한다. 본 연구는 폰트 클래스의 독립성을 유지하면서도, 렌더링 이미지에서 나타나는 클래스 간 시각적 유사성이 폰트 인식 학습에 미치는 영향을 문제의 중심으로 설정한다.

### 선행 연구

폰트 인식은 렌더링된 문자 이미지로부터 해당 글꼴의 클래스를 추정하는 세밀한 시각 분류 문제로 다루어져 왔다. 대표적으로 DeepFont는 Visual Font Recognition 문제를 정의하고, 합성 데이터와 실제 텍스트 이미지를 함께 활용하는 대규모 AdobeVFR 데이터셋을 구축하였다 [1]. 또한 합성 데이터와 실제 이미지 간의 domain mismatch를 완화하기 위해 CNN 기반 모델과 stacked convolutional auto-encoder를 이용한 domain adaptation을 적용하였으며, 폰트 식별뿐 아니라 유사 폰트 추천을 위한 similarity measure도 함께 제시하였다 [1]. 이러한 연구는 딥러닝 기반 폰트 인식의 가능성을 보였으나, 기본적으로 폰트 이름을 기준으로 한 단일 클래스 분류 문제에 초점을 둔다.

이후 CNN을 이용한 폰트 분류 연구들은 특정 언어권 또는 문서 이미지 환경에서 폰트 클래스를 분류하는 방향으로 확장되었다. Tensmeyer 등은 작은 텍스트 패치를 CNN으로 분류한 뒤 패치별 예측을 평균하여 문서나 줄 단위의 폰트 클래스를 추정하는 방법을 제안하였고, 아랍어 폰트 및 중세 라틴 필사본 분류에서 높은 성능을 보고하였다 [2]. 최근 Persis는 페르시아어 폰트 인식을 위한 공개 데이터셋과 CNN 기반 파이프라인을 제안하여, 별도의 수작업 특징 추출 없이도 폰트 인식이 가능함을 보였다 [3]. 이들 연구는 폰트 인식 성능 향상과 데이터셋 구축에 기여했지만, 대부분 정답 폰트 하나만을 양성 클래스로 두고 나머지 폰트를 모두 동일한 음성 클래스로 간주한다는 점에서, 시각적으로 매우 유사한 폰트 클래스 간의 모호성을 명시적으로 다루지는 않는다.

단일 hard label의 한계를 완화하기 위한 연구도 널리 수행되어 왔다. Label smoothing은 정답 클래스에 모든 확률 질량을 집중시키는 one-hot target 대신, 일부 확률을 다른 클래스에 분산시켜 모델의 과신을 줄이는 대표적인 정규화 방법이다 [4]. Knowledge distillation 역시 teacher model의 soft target을 이용해 hard label보다 풍부한 클래스 간 정보를 student model에 전달한다 [5]. 이러한 soft target 기반 접근은 클래스 간 관계를 반영할 수 있다는 장점이 있으나, 일반적으로 사전 학습된 teacher model이나 고정된 smoothing 규칙에 의존한다.

클래스 간 유사성 또는 라벨의 불확실성을 직접 다루려는 연구들도 본 연구와 관련된다. Reed 등은 noisy label 환경에서 모델 예측과 라벨 정보를 결합하는 bootstrapping 방식으로 불완전하거나 주관적인 라벨에 대한 강건성을 높이고자 하였다 [6]. Sukhbaatar 등은 noisy label 분포를 모델링하기 위해 네트워크에 noise layer를 추가하는 방법을 제안하였다 [7]. 이러한 접근들은 라벨이 항상 명확하고 배타적이지 않다는 점을 반영한다는 점에서 본 연구와 문제의식이 유사하다. 그러나 많은 경우 외부적으로 정의된 confidence label, noise model, 또는 사전 계산된 클래스 유사도에 의존하며, 폰트 인식처럼 클래스 수가 많고 시각적 유사성이 샘플별로 달라질 수 있는 문제에서는 적용이 제한될 수 있다.

학습 과정에서 중요한 음성 샘플을 선택하거나 손실의 기여도를 조절하는 연구도 관련된다. Focal Loss는 dense object detection에서 다수의 쉬운 음성 예제가 학습을 지배하는 문제를 해결하기 위해, 잘 분류된 예제의 손실을 낮추고 어려운 예제에 더 집중하도록 cross-entropy를 변형하였다 [8]. 이는 음성 클래스 또는 음성 샘플을 모두 동일하게 취급하지 않는다는 점에서 본 연구와 연결된다. 다만 Focal Loss는 어려운 음성 예제에 더 큰 가중치를 부여하는 방향인 반면, 본 연구는 모델이 현재 시각적으로 혼동 가능하다고 판단한 상위 $k$개의 비정답 클래스를 negative loss에서 일시적으로 제외하여, 유사 클래스에 대한 과도한 억제를 완화한다는 점에서 차이가 있다.

Curriculum learning은 학습 초기에는 상대적으로 쉬운 조건에서 모델을 안정화한 뒤, 점차 더 어려운 학습 목표로 전환하는 전략이다 [9]. 본 연구 역시 학습 초기에는 모델의 top-k 예측이 신뢰하기 어렵다는 점을 고려하여, 먼저 일반적인 단일 라벨 기반 warm-up 학습을 수행한 뒤 Top-k Relaxed Negative Loss의 비중을 점진적으로 증가시키는 curriculum 전략을 사용한다. 이를 통해 초기 학습의 불안정성을 줄이면서, 모델 출력이 점차 의미 있는 시각적 유사성 구조를 반영하도록 유도한다.

종합하면, 기존 폰트 인식 연구는 주로 CNN 기반 특징 학습, 데이터셋 구축, domain adaptation, 또는 언어권별 폰트 분류 성능 향상에 초점을 맞추어 왔다 [1, 2, 3]. 반면 soft label, noisy label, hard example mining, curriculum learning 관련 연구들은 단일 hard label 학습의 한계를 완화하는 다양한 방향을 제시하였다 [4, 5, 6, 7, 8, 9]. 본 연구는 이러한 흐름을 폰트 인식 문제에 맞게 결합하되, 사전 학습된 폰트 인코더나 외부 유사도 행렬 없이 모델의 현재 sigmoid activation만을 이용해 ambiguous candidate를 동적으로 선택한다. 선택된 후보는 pseudo-label로 강제되지 않고 negative penalty에서 제외되는 완화 대상으로만 사용되므로, 폰트 이름 기반 라벨 체계를 유지하면서도 시각적으로 유사한 클래스에 대한 약한 multi-hot activation을 허용할 수 있다.

## 2. 본론

본 연구에서는 폰트 인식 문제에서 발생하는 클래스 간 시각적 모호성을 완화하기 위해, 기존 단일 클래스 기반 학습 목표를 수정한 **동적 후보 선택 기반 Top-k Relaxed Negative Learning**을 제안한다. 제안 방법은 사전 학습된 인코더나 외부 유사도 행렬을 필요로 하지 않으며, 학습 과정에서 모델의 현재 출력값을 이용하여 정답 클래스 외에 시각적으로 유사할 가능성이 있는 후보 클래스를 동적으로 선택한다. 이를 통해 모델은 정답 폰트 클래스에 대한 식별력을 유지하면서도 일부 모호한 비정답 클래스에 대해 multi-hot activation을 허용할 수 있다.

### 2.1 기존 단일 라벨 학습의 한계

전체 폰트 클래스 수를 $C$, 입력 이미지와 라벨을 각각 $x_i$, $y_i$라고 하자. 모델은 입력 이미지 $x_i$에 대해 클래스별 logit 벡터를 출력한다.

$$
\mathbf{z}_i = f_\theta(x_i) \in \mathbb{R}^{C}
$$

기존 softmax 기반 cross-entropy 학습에서는 정답 클래스 $y_i$만을 양성 클래스로 간주하고, 나머지 $C-1$개의 클래스는 모두 동일한 음성 클래스로 취급한다. 그러나 폰트 인식 문제에서는 서로 다른 폰트 이름을 갖는 클래스라 하더라도 렌더링된 이미지 상에서 매우 유사한 형태를 보일 수 있다. 특히 특정 문자, 낮은 해상도, 제한된 이미지 크기와 같은 조건에서는 정답 폰트와 시각적으로 유사한 다른 폰트 간의 구분이 본질적으로 모호할 수 있다.

이러한 상황에서 모든 비정답 클래스를 동일하게 억제하는 학습 방식은 지나치게 경직된 지도 신호를 제공한다. 모델이 정답 클래스와 시각적으로 유사한 다른 폰트 클래스에 대해 높은 응답을 보이더라도, 기존 학습 목표에서는 이를 완전한 오답으로 간주한다. 결과적으로 모델은 실제 시각적 유사성 구조를 반영하기보다 폰트 이름 기반 클래스의 배타성을 과도하게 학습하게 된다.

본 연구는 이러한 문제를 해결하기 위해 softmax 대신 sigmoid 기반 출력을 사용한다. 각 클래스의 예측값은 다음과 같이 정의된다.

$$
p_{i,c} = \sigma(z_{i,c})
$$

여기서 $p_{i,c} \in [0,1]$은 입력 $x_i$가 클래스 $c$에 대해 갖는 독립적인 activation score를 의미한다. sigmoid 출력은 softmax와 달리 클래스 간 합이 1로 제한되지 않으므로, 하나의 입력 이미지에 대해 여러 클래스가 동시에 높은 activation을 가질 수 있다.

### 2.2 동적 ambiguous candidate 선택

본 연구의 핵심은 사전 정의된 클래스 유사도 없이도 모델이 학습 과정에서 스스로 모호한 후보 클래스를 선택하도록 하는 것이다. 기존 similarity-aware multi-hot 학습에서는 클래스 간 유사도 $\mathrm{sim}(a,b)$를 계산한 뒤, 정답 클래스와 유사한 클래스 집합을 미리 구성한다. 그러나 실제 환경에서는 사전 학습된 폰트 인코더나 신뢰할 수 있는 유사도 행렬이 존재하지 않을 수 있다. 따라서 본 연구에서는 외부 유사도 정보 대신 모델의 현재 출력값을 이용하여 ambiguous candidate를 동적으로 정의한다.

입력 $x_i$에 대해 정답 클래스를 제외한 클래스 중 activation score가 높은 상위 $k$개 클래스를 다음과 같이 정의한다.

$$
\mathcal{A}_i = \operatorname{TopK}_{c \neq y_i} \left( p_{i,c} \right)
$$

여기서 $\mathcal{A}_i$는 입력 $x_i$에 대한 ambiguous candidate set을 의미한다. 이 집합에 포함된 클래스들은 현재 모델이 정답은 아니지만 시각적으로 유사하거나 혼동 가능성이 있다고 판단한 클래스들이다. 즉, $\mathcal{A}_i$는 고정된 외부 지식에 의해 결정되는 것이 아니라, 매 학습 단계에서 모델의 출력에 따라 동적으로 갱신된다.

이러한 동적 후보 선택 방식은 다음과 같은 장점을 갖는다. 첫째, 사전 학습된 모델이나 별도의 유사도 계산 과정이 필요하지 않다. 둘째, 학습이 진행됨에 따라 모델의 표현 공간이 개선되면 ambiguous candidate 역시 점진적으로 더 의미 있는 클래스들로 변화할 수 있다. 셋째, 각 입력 이미지마다 관측되는 문자 형태와 난이도가 다르기 때문에 샘플별로 서로 다른 후보 집합을 구성할 수 있다.

### 2.3 Top-$k$ Relaxed Negative Loss

동적으로 선택된 ambiguous candidate는 명시적인 정답 클래스는 아니지만 시각적으로 유사할 가능성이 있는 클래스이다. 따라서 본 연구에서는 이들에 대해 일반적인 음성 클래스와 동일한 수준의 negative penalty를 부과하지 않는다. 구체적으로, 정답 클래스 $y_i$와 ambiguous candidate set $\mathcal{A}_i$를 제외한 클래스만을 relaxed negative class로 정의한다.

$$
\mathcal{N}_i^{\mathrm{relaxed}} = \{1,\ldots,C\} \setminus \left( \{y_i\} \cup \mathcal{A}_i \right)
$$

$\mathcal{N}_i^{\mathrm{relaxed}}$에 포함된 클래스들은 현재 모델 관점에서 정답 클래스와 관련성이 낮은 일반적인 음성 클래스이다. 따라서 이들에 대해서는 activation이 낮아지도록 negative loss를 적용한다.

정답 클래스에 대해서는 다음과 같은 positive loss를 사용한다.

$$
\mathcal{L}_{\mathrm{pos}}(i) = -\log p_{i,y_i} = -\log \sigma(z_{i,y_i})
$$

이는 원래 데이터셋의 단일 폰트 라벨을 보존하기 위한 항이다. 즉, multi-hot activation을 허용하더라도 실제 라벨 클래스 $y_i$는 가장 중요한 supervision으로 유지된다.

반면 relaxed negative class에 대해서는 다음과 같은 negative loss를 적용한다.

$$
\mathcal{L}_{\mathrm{neg}}(i) = - \frac{1}{ \left| \mathcal{N}_i^{\mathrm{relaxed}} \right| } \sum_{c \in \mathcal{N}_i^{\mathrm{relaxed}}} \log \left( 1-p_{i,c} \right)
$$

이 항은 정답 클래스와 ambiguous candidate를 제외한 대부분의 무관한 클래스가 낮은 activation을 갖도록 유도한다. 중요한 점은 ambiguous candidate set $\mathcal{A}_i$에 포함된 클래스는 이 negative loss에서 제외된다는 것이다. 따라서 모델은 정답 클래스 이외에도 상위 $k$개의 후보 클래스에 대해 높은 activation을 유지할 수 있다.

다만 ambiguous candidate에 대해 아무런 제약을 두지 않을 경우, 모델이 지나치게 많은 비정답 클래스에 대해 높은 activation을 출력하거나 top-k 후보의 activation을 무분별하게 증가시키는 문제가 발생할 수 있다. 이를 방지하기 위해 본 연구에서는 정답 클래스를 제외한 전체 비정답 클래스에 대해 약한 sparsity regularization을 추가한다.

$$
\mathcal{L}_{\mathrm{sparse}}(i) = \sum_{c \neq y_i} p_{i,c}
$$

이 항은 비정답 클래스 activation의 총량을 제한함으로써 모델이 모든 클래스를 동시에 활성화하는 trivial solution으로 수렴하는 것을 방지한다. 단, 본 연구의 목적은 일부 유사 클래스의 activation을 허용하는 것이므로 sparsity regularization의 가중치는 negative loss에 비해 작게 설정한다.

최종적으로 제안하는 Top-k Relaxed Negative Loss는 다음과 같이 정의된다.

$$
\mathcal{L}_{\mathrm{TRN}}(i) = \mathcal{L}_{\mathrm{pos}}(i) + \lambda \mathcal{L}_{\mathrm{neg}}(i) + \beta \mathcal{L}_{\mathrm{sparse}}(i)
$$

즉,

$$
\mathcal{L}_{\mathrm{TRN}}(i) = -\log \sigma(z_{i,y_i}) - \lambda \frac{1}{ \left| \mathcal{N}_i^{\mathrm{relaxed}} \right| } \sum_{c \in \mathcal{N}_i^{\mathrm{relaxed}}} \log \left( 1-\sigma(z_{i,c}) \right) + \beta \sum_{c \neq y_i} \sigma(z_{i,c})
$$

여기서 $\lambda$는 relaxed negative loss의 강도를 조절하는 하이퍼파라미터이고, $\beta$는 sparsity regularization의 강도를 조절하는 하이퍼파라미터이다. 일반적으로 $\beta$는 $\lambda$보다 작게 설정하여 multi-hot activation의 가능성을 유지하면서도 과도한 activation 확산만을 억제한다.

### 2.4 Curriculum 기반 학습 전략

Top-k Relaxed Negative Loss는 모델의 현재 출력값을 이용해 ambiguous candidate를 선택한다. 그러나 학습 초기에는 모델의 예측이 충분히 안정적이지 않기 때문에 초기 top-k 후보가 실제로 시각적으로 유사한 클래스라고 보기 어렵다. 이 시점에서 바로 negative relaxation을 적용하면 우연히 높은 activation을 보인 잘못된 클래스가 ambiguous candidate로 보호될 수 있으며, 이는 학습의 불안정성을 증가시킬 수 있다.

이를 해결하기 위해 본 연구에서는 curriculum learning 전략을 도입한다. 학습 초기에는 일반적인 단일 라벨 학습을 통해 모델이 기본적인 폰트 식별 능력을 먼저 획득하도록 한다. 이후 모델의 예측이 일정 수준 안정화되면 점진적으로 Top-k Relaxed Negative Loss의 비중을 증가시킨다.

초기 warm-up 단계에서는 다음과 같은 기본 sigmoid 기반 binary loss를 사용한다.

$$
\mathcal{L}_{\mathrm{warm}}(i) = -\log \sigma(z_{i,y_i}) - \lambda \frac{1}{C-1} \sum_{c \neq y_i} \log \left( 1-\sigma(z_{i,c}) \right)
$$

이 단계에서는 모든 비정답 클래스를 음성 클래스로 간주한다. 이는 모델이 최소한의 클래스 구분 능력을 형성하도록 하기 위한 단계이다.

이후 curriculum 단계에서는 warm-up loss와 Top-k Relaxed Negative Loss를 선형 결합한다.

$$
\mathcal{L}_{\mathrm{curr}}(i,t) = \left( 1-\alpha_t \right) \mathcal{L}_{\mathrm{warm}}(i) + \alpha_t \mathcal{L}_{\mathrm{TRN}}(i)
$$

여기서 $t$는 현재 epoch 또는 iteration을 의미하며, $\alpha_t$는 relaxed learning의 적용 비율을 조절하는 curriculum coefficient이다. 본 연구에서는 다음과 같은 선형 증가 스케줄을 사용할 수 있다.

$$
\alpha_t = \min \left( 1, \frac { t-T_{\mathrm{warm}} } { T_{\mathrm{ramp}} } \right)
$$

단, $t < T_{\mathrm{warm}}$인 경우에는 다음과 같이 둔다.

$$
\alpha_t = 0
$$

즉, $T_{\mathrm{warm}}$까지는 warm-up loss만 사용하고 이후 $T_{\mathrm{ramp}}$ 동안 Top-k Relaxed Negative Loss의 비중을 점진적으로 증가시킨다. 최종적으로 $\alpha_t=1$이 되면 전체 학습 목표는 완전히 $\mathcal{L}_{\mathrm{TRN}}$으로 전환된다.

이러한 curriculum 전략은 동적 후보 선택의 신뢰도를 점진적으로 높이는 역할을 한다. 학습 초반에는 모델이 아직 유사 폰트를 구분할 수 있는 표현을 갖추지 못했으므로 엄격한 단일 라벨 supervision을 사용한다. 반면 학습이 진행된 후에는 모델의 출력 분포가 점차 의미 있는 구조를 갖게 되므로 top-k 후보를 ambiguous candidate로 간주하고 negative penalty를 완화한다.

### 2.5 학습 절차

제안 방법의 학습 절차는 다음과 같다. 먼저 입력 이미지 $x_i$를 모델에 통과시켜 클래스별 logit $\mathbf{z}_i$를 얻고, sigmoid 함수를 통해 class-wise activation $p_{i,c}$를 계산한다. 학습 초기 warm-up 구간에서는 정답 클래스만 positive로 두고 나머지 모든 클래스를 negative로 두어 기본적인 식별 능력을 학습한다.

warm-up 이후에는 정답 클래스 $y_i$를 제외한 클래스 중 activation score가 가장 높은 상위 $k$개 클래스를 선택하여 ambiguous candidate set $\mathcal{A}_i$를 구성한다. 이 후보 집합은 매 iteration마다 현재 모델의 출력에 따라 갱신된다. 이후 정답 클래스와 ambiguous candidate를 제외한 나머지 클래스를 relaxed negative class $\mathcal{N}_i^{\mathrm{relaxed}}$로 정의하고, 이들에 대해서만 negative loss를 적용한다. 동시에 전체 비정답 클래스 activation에 대해 약한 sparsity regularization을 부여하여 과도한 multi-hot activation을 방지한다.

이를 정리하면 다음과 같다.

$$
\mathbf{z}_i = f_\theta(x_i)
$$

$$
p_{i,c} = \sigma(z_{i,c})
$$

$$
\mathcal{A}_i = \operatorname{TopK}_{c \neq y_i} \left( p_{i,c} \right)
$$

$$
\mathcal{N}_i^{\mathrm{relaxed}} = \{1,\ldots,C\} \setminus \left( \{y_i\} \cup \mathcal{A}_i \right)
$$

$$
\mathcal{L}_{\mathrm{curr}}(i,t) = \left( 1-\alpha_t \right) \mathcal{L}_{\mathrm{warm}}(i) + \alpha_t \mathcal{L}_{\mathrm{TRN}}(i)
$$

이 과정에서 ambiguous candidate는 명시적인 pseudo-label로 고정되지 않는다. 즉, $\mathcal{A}_i$는 정답으로 강제되는 클래스가 아니라 negative penalty에서 일시적으로 제외되는 완화 대상이다. 따라서 제안 방법은 noisy pseudo-labeling과 달리 잘못된 후보 클래스를 직접 positive로 학습시키는 위험을 줄인다. 동시에 기존 단일 라벨 학습과 달리 시각적으로 유사할 가능성이 있는 클래스까지 강하게 억제하지 않음으로써 폰트 인식 문제의 모호성을 보다 유연하게 반영한다.

### 2.6 제안 방법의 효과

제안하는 Top-k Relaxed Negative Learning은 폰트 인식 문제의 세 가지 요구를 동시에 만족한다. 첫째, 정답 클래스에 대한 positive loss를 유지하므로 데이터셋의 폰트 이름 기반 라벨 체계는 보존된다. 둘째, 모델 출력의 상위 $k$개 비정답 클래스를 ambiguous candidate로 선택하고 negative loss에서 제외함으로써 시각적으로 유사한 폰트 클래스에 대한 multi-hot activation을 허용한다. 셋째, 나머지 대부분의 클래스에는 negative loss를 적용하고 sparsity regularization을 추가함으로써 모든 클래스가 동시에 활성화되는 collapse를 방지한다.

또한 제안 방법은 외부 유사도 계산에 의존하지 않는다. 클래스 간 시각적 유사도는 사전 정의되는 것이 아니라 모델의 학습 과정에서 출력 분포를 통해 암묵적으로 추정된다. 따라서 사전 학습된 폰트 인코더가 없거나 클래스 간 유사도 행렬을 신뢰하기 어려운 상황에서도 적용 가능하다.

결과적으로 본 방법은 단일 라벨 폰트 인식 문제를 완전한 다중 클래스 분류로만 취급하지 않고, 시각적으로 모호한 후보 클래스가 존재할 수 있는 약한 multi-hot recognition 문제로 확장한다. 이를 통해 모델은 정답 클래스의 식별력을 유지하면서도 실제 렌더링 이미지에서 나타나는 폰트 간 시각적 유사성과 불확실성을 보다 자연스럽게 반영할 수 있다.

## 실험 결과

### 3.1 실험 로그

본 절의 비교는 아래 두 학습 로그를 기준으로 한다.

- baseline: `data/checkpoints-2-interleaved/v1/metrics.jsonl`
- 제안 방법: `data/checkpoints-2-interleaved/v2/metrics.jsonl`

baseline은 softmax cross-entropy 기반 단일 라벨 분류 학습이며, 제안 방법은
본 문서 2.3~2.4절의 sigmoid 기반 Top-$k$ Relaxed Negative Learning 구현 로그다.
두 로그 모두 **FontHead 강화 적용 이후의 모델**로 학습한 결과이며, 따라서
아래 비교는 "기존 약한 FontHead vs 강화된 FontHead" 비교가 아니라
**강화된 동일 계열 모델에서 baseline 손실과 제안 손실을 비교한 결과**로
해석해야 한다. 다만 실제 로그를 확인하면 두 실험의 총 학습 길이는 서로 다르다.

- baseline 로그: epoch 0~2, 총 74,250 step
- 제안 방법 로그: epoch 0~1, 총 50,950 step

따라서 최종 step끼리의 직접 비교는 완전히 공정하지 않다. 이를 보완하기 위해
본 절에서는 (1) 동일 step 부근 비교, (2) 각 실험의 최고 성능 비교,
(3) 마지막 기록 비교를 함께 제시한다.

### 3.2 정량 비교

#### 동일 step 비교

| 비교 시점 | 방법 | font top-1 acc | font top-5 acc | syllable acc | total loss | jamo loss | font loss | 비고 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| step 32,000 | baseline | 0.993646 | 0.999896 | 0.989896 | 0.059736 | 0.039846 | 0.019889 | epoch 0 |
| step 32,000 | 제안 방법 | 0.195833 | 0.735000 | 0.988438 | 0.070754 | 0.046140 | 0.024614 | epoch 0, `alpha=0.0` |
| step 50,000 부근 | baseline | 0.978750 | 0.999375 | 0.983125 | 0.126844 | 0.060179 | 0.066666 | step 50,000, epoch 1 |
| step 50,000 부근 | 제안 방법 | 0.091797 | 0.418438 | 0.994531 | 0.096237 | 0.030344 | 0.065893 | step 50,000, epoch 1, `alpha=0.0` |

동일 step 기준으로 보면, 제안 방법은 baseline보다 **jamo 정확도는 비슷하거나 더 높지만**,
폰트 top-1 / top-5 정확도는 큰 폭으로 낮다. 특히 step 32,000에서 baseline의
font top-1 정확도는 이미 99%를 넘는 반면, 제안 방법은 19.6% 수준에 머문다.
step 50,000 부근에서는 이 격차가 더 커져 top-1 정확도가 약 9.2%까지 하락한다.

#### 최고 성능 비교

| 지표 | baseline | 제안 방법 | 관찰 |
| --- | ---: | ---: | --- |
| 최고 font top-1 acc | 0.995208 | 0.203437 | baseline이 압도적으로 높다 |
| 최고 font top-5 acc | 1.000000 | 0.735000 | 제안 방법은 top-5도 1.0에 근접하지 못했다 |
| 최고 syllable acc | 0.996667 | 0.996719 | jamo/음절 인식은 거의 비슷하다 |
| 최소 total loss | 0.028135 | 0.055682 | baseline이 더 낮다 |
| 최소 jamo loss | 0.012533 | 0.013996 | 두 방법 모두 충분히 낮다 |

최고 성능 기준으로도 결과는 동일하다. 제안 방법은 jamo 인식 자체는 baseline과
비슷한 수준까지 도달했지만, 폰트 인식에서는 baseline을 전혀 따라가지 못했다.
즉 본 실험에서는 제안한 relaxed learning이 "폰트 유사성 완화"로 이어지지
않았고, 오히려 폰트 식별력 자체를 심하게 약화시켰다.

#### 마지막 기록 비교

마지막 기록은 아래와 같다.

- baseline: step 74,250에서 `font_acc=0.984688`, `font_top5_acc=0.999844`
- 제안 방법: step 50,950에서 `font_acc=0.021406`, `font_top5_acc=0.091172`

총 학습 길이가 다르므로 이 비교만으로 결론을 내려서는 안 된다. 그러나 제안
방법 쪽은 마지막 구간으로 갈수록 font top-1 / top-5 정확도가 지속적으로
악화되어, 추가 학습 없이도 불안정성이 분명하게 드러난다.

### 3.3 핵심 관찰

이번 실험에서 확인된 현상은 다음과 같다.

1. **baseline은 현재 데이터셋에서 매우 강한 기준선이다.**
   동일 step 기준으로도, 최고 성능 기준으로도 font top-1 / top-5 정확도는
   baseline이 압도적으로 높았다.

2. **제안 방법은 jamo 학습을 방해하지 않았지만 font 학습을 방해했다.**
   syllable accuracy는 baseline과 비슷하거나 약간 더 높게 나오는 구간도 있었지만,
   font accuracy는 크게 낮았다. 이는 모델이 글자 구조 정보는 잘 학습했으나,
   폰트를 하나의 날카로운 단일 클래스로 식별하는 능력은 잃었다는 뜻이다.

3. **제안 방법의 loss 값은 정확도와 일치하지 않았다.**
   예를 들어 step 50,000 부근에서 제안 방법의 total loss와 font loss는
   baseline보다 비슷하거나 더 낮아 보이지만, font top-1 정확도는 훨씬 낮다.
   따라서 이 실험에서는 loss의 절대값만으로 성능을 해석하면 안 되며,
   실제 비교 지표는 font top-1 / top-5 정확도여야 한다.

4. **가장 중요한 사실은, 이번 v2 로그에서 curriculum이 실제로 켜지지 않았다.**
   `v2/metrics.jsonl`의 `alpha` 값은 로그 전체에서 끝까지 **0.0**이었다.
   즉 본 문서 2.4절에서 제안한 "warm-up 이후 점진적으로 relaxed loss를 켠다"는
   curriculum이 이번 실험에서는 한 번도 활성화되지 않았다. 엄밀히 말하면 이
   로그는 Top-$k$ Relaxed Negative Learning 전체를 검증한 실험이라기보다,
   **warm-up sigmoid objective만으로 학습한 실험**에 가깝다.

### 3.4 원인 분석

이번 실험이 baseline보다 나쁜 결과를 낸 원인은 크게 네 가지로 해석할 수 있다.

#### (1) curriculum이 끝까지 warm-up에 머물렀다

가장 직접적인 원인은 `alpha=0.0`이 끝까지 유지되었다는 점이다.
본 문서 2.4절의 설계 의도는, 초기에 `L_warm`으로 안정화한 뒤 이후
`L_TRN`의 비중을 점차 늘리는 것이다. 그러나 이번 로그에서는 학습이 epoch 2에
도달하기 전에 종료되어, 실제 relaxed negative phase가 시작되지 않았다.
따라서 이번 결과는 제안 방법의 핵심 아이디어 자체를 충분히 시험하지 못했다.

즉, "제안 방법의 전체 curriculum이 실패했다"기보다, **실험이 warm-up 단계에
머문 채 종료되었고 그 warm-up objective가 baseline보다 나빴다**고 보는 것이
더 정확하다.

#### (2) sigmoid 기반 독립 클래스 학습이 단일 정답 분류 문제와 잘 맞지 않았다

현재 폰트 인식 문제는 실제 평가 시점에서 하나의 정답 폰트를 맞히는
**single-label classification** 문제이다. baseline의 softmax cross-entropy는
이 목표와 정확히 일치하며, 정답 클래스를 올리고 다른 모든 클래스를 상대적으로
내리는 경쟁 구조를 제공한다.

반면 warm-up sigmoid loss는 각 클래스를 더 독립적으로 다루기 때문에,
정답 클래스 하나를 날카롭게 선택하는 압력이 softmax보다 약하다. 그 결과
시각적으로 비슷한 폰트들이 동시에 어느 정도 activation을 유지할 수 있고,
이는 top-5 정확도는 어느 정도 남기면서도 top-1 정확도를 크게 떨어뜨리는
방향으로 작용할 수 있다. step 32,000에서 top-5는 0.735까지 올라가지만
top-1은 0.196에 그친 것이 이런 현상을 잘 보여 준다.

#### (3) 폰트 손실의 절대 스케일이 baseline과 다르다

제안 방법의 font loss는 sigmoid 기반 양성/음성 항의 평균으로 정의되므로,
softmax CE를 쓰는 baseline의 font loss와 **절대 스케일이 다르다**. 따라서
제안 방법의 font loss가 더 작게 보인다고 해서 실제 분류 성능이 더 좋다는
뜻은 아니다. 이번 실험에서 step 50,000 부근의 `font_loss`는 두 방법이 거의
비슷하지만, font top-1 정확도는 0.978750 대 0.091797로 극단적으로 다르다.
즉 제안 방법은 손실 정의상 낮은 값을 만들 수는 있었지만, 평가 목표인
정답 폰트 식별력으로 연결되지 못했다.

### 3.5 정리

이번 실험에서는 본 문서의 제안 방법이 baseline을 개선하지 못했으며, 실제로는
font 인식 성능을 크게 악화시켰다. 다만 `alpha`가 끝까지 0.0이었다는 점에서,
이번 로그는 Top-$k$ Relaxed Negative Learning의 완전한 평가라기보다
"sigmoid warm-up objective의 실패 사례"로 해석하는 것이 타당하다.

따라서 현 시점의 결론은 다음과 같다.

- 현재 구현/설정 기준으로는 baseline을 유지하는 것이 옳다.
- 제안 방법을 다시 검증하려면, 최소한 warm-up 이후 `alpha>0` 구간까지 실제로
  진입하는 실험을 별도로 수행해야 한다.
- 다음 검증은 loss 설계와 curriculum 스케줄을 직접 재검토하는
  방향이어야 한다.

## 결론

본 문서가 제안한 동적 후보 선택 기반 Top-$k$ Relaxed Negative Learning은,
현재 구현과 실험 설정에서는 baseline softmax 학습을 이기지 못했다. 실제 로그
비교 결과, jamo 인식은 유지되었지만 font top-1 / top-5 정확도는 크게 악화되었다.
또한 이번 실험에서는 curriculum의 `alpha`가 끝까지 0.0이어서 relaxed phase가
실제로 활성화되지 않았으므로, 현재 결과는 제안 방법 전체의 성공 사례가 아니라
warm-up objective의 실패 사례로 해석하는 것이 맞다. 또한 이 결론은 이미
FontHead를 강화한 모델에서도 유지되므로, 향후 재검증이 필요하다면 헤드 구조를
또 키우는 것보다 `alpha>0` 구간까지 실제로 진입하는 실험을 다시 수행하고,
loss 설계와 curriculum 스케줄 자체를 직접 점검하는 것이 우선이다.

## References

1. Zhangyang Wang, Jianchao Yang, Hailin Jin, Eli Shechtman, Aseem Agarwala, Jonathan Brandt, Thomas S. Huang, DeepFont: Identify Your Font from An Image, ACM Multimedia, 2015.

2. Chris Tensmeyer, Daniel Saunders, Tony Martinez, Convolutional Neural Networks for Font Classification, International Conference on Document Analysis and Recognition, 2017.

3. Mehrdad Mohammadian, Neda Maleki, Tobias Olsson, Fredrik Ahlgren, Persis: A Persian Font Recognition Pipeline Using Convolutional Neural Networks, arXiv, 2023.

4. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, Rethinking the Inception Architecture for Computer Vision, IEEE Conference on Computer Vision and Pattern Recognition, 2016.

5. Geoffrey Hinton, Oriol Vinyals, Jeff Dean, Distilling the Knowledge in a Neural Network, arXiv, 2015.

6. Scott Reed, Honglak Lee, Dragomir Anguelov, Christian Szegedy, Dumitru Erhan, Andrew Rabinovich, Training Deep Neural Networks on Noisy Labels with Bootstrapping, arXiv, 2014.

7. Sainbayar Sukhbaatar, Joan Bruna, Manohar Paluri, Lubomir Bourdev, Rob Fergus, Training Convolutional Networks with Noisy Labels, arXiv, 2014.

8. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár, Focal Loss for Dense Object Detection, IEEE International Conference on Computer Vision, 2017.

9. Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston, Curriculum Learning, International Conference on Machine Learning, 2009.
