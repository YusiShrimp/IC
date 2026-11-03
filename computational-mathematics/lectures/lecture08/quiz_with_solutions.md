# Computational Mathematics -- Lecture 8 quiz solutions

$\newcommand{\ii}{\mathrm{i}}\newcommand{\dd}{\mathrm{d}}\newcommand{\rr}{\mathbb{R}}\newcommand{\cc}{\mathbb{C}}$

### Question 1

For the dataset $\{1, 3, 7, 13\}$, compute the variance.

- **(A)** $21$
- **(B)** $28$
- **(C)** $57$
- **(D)** $4$

> **Answer: (A).**
>
> Mean $=(1+3+7+13)/4=6$. Population variance $=\big((1-6)^2+(3-6)^2+(7-6)^2+(13-6)^2\big)/4 = (25+9+1+49)/4 = 21$. The $28$ option divides by $n-1=3$ instead; the $4$ option averages the absolute deviations $(5+3+1+7)/4=4$ instead of squaring them; the $57$ option reports the mean of the squares $(1+9+49+169)/4=57$ without subtracting the squared mean $6^2=36$.

---

### Question 2

The data set $\{1, 3, 4, 5, 7\}$ is the whole population of interest. What is the z-score of the value 7?

- **(A)** $3.5$
- **(B)** $1.34$
- **(C)** $0.75$
- **(D)** $1.5$

> **Answer: (D).**
>
> The mean is 4 and the population variance is $\frac{9+1+0+1+9}{5} = 4$, so $\sigma = 2$ and $z = \frac{7-4}{2} = 1.5$: the value 7 lies 1.5 standard deviations above the mean. Dividing by the variance gives 0.75; dividing 7 itself by $\sigma$ gives 3.5; and the sample standard deviation $\sqrt{20/4} = \sqrt5$ gives $3/\sqrt5 \approx 1.34$.

---

### Question 3

Treating the paired data $x = (1, 2, 3, 4)$ and $y = (2, 4, 5, 9)$ as the whole population, what is the covariance $\mathrm{Cov}(x, y)$?

- **(A)** $\tfrac{61}{4}$
- **(B)** $\tfrac{11}{4}$
- **(C)** $\tfrac{11}{3}$
- **(D)** $11$

> **Answer: (B).**
>
> The means are $\bar x = \frac52$ and $\bar y = 5$. The deviations are $(-\frac32, -\frac12, \frac12, \frac32)$ and $(-3, -1, 0, 4)$, and their products sum to $4.5 + 0.5 + 0 + 6 = 11$. The population covariance divides by $n = 4$: $\mathrm{Cov}(x, y) = \frac{11}{4}$. Dividing by $n - 1 = 3$ gives the sample covariance $\frac{11}{3}$; averaging $x_i y_i$ without subtracting the means gives $\frac{61}{4}$; and 11 is the sum before averaging.

---

### Question 4

Two fair six-sided dice are rolled. What is the probability that the sum of the two numbers is 8 (to 3 decimal places)?

- **(A)** $0.139$
- **(B)** $0.091$
- **(C)** $0.143$
- **(D)** $0.083$

> **Answer: (A).**
>
> The 36 ordered outcomes $(a, b)$ are equally likely, and five of them sum to 8: $(2,6)$, $(3,5)$, $(4,4)$, $(5,3)$ and $(6,2)$. So $P = \frac{5}{36} \approx 0.139$. The 11 possible sums are not equally likely, so $\frac{1}{11} \approx 0.091$ is wrong. Counting only the unordered pairs $\{2,6\}$, $\{3,5\}$, $\{4,4\}$ gives $\frac{3}{36} \approx 0.083$ over the ordered outcomes, or $\frac{3}{21} \approx 0.143$ over the 21 unordered ones, which are not equally likely either.

---

### Question 5

A random variable $X$ has probability density function $p(x) = c\,(4 - x)$ for $0 \le x \le 4$, and $p(x) = 0$ otherwise. What is the value of the constant $c$?

- **(A)** $\tfrac{1}{24}$
- **(B)** $\tfrac{1}{16}$
- **(C)** $\tfrac{1}{8}$
- **(D)** $\tfrac{1}{4}$

> **Answer: (C).**
>
> The total area under a PDF must be 1: $\int_0^4 c(4-x)\,dx = c\left[4x - \frac{x^2}{2}\right]_0^4 = c(16 - 8) = 8c$, so $c = \frac18$. (This is the area of a triangle with base 4 and height $4c$.) Making the peak value $p(0) = 4c$ equal to 1 gives $\frac14$, but a density is not a probability and may exceed 1. Dropping the $-\frac{x^2}{2}$ term gives $\frac{1}{16}$, and adding it instead gives $\frac{1}{24}$.

---

### Question 6

The sepal length $X$ (in cm) of a species of iris is modelled as normal with mean $\mu = 5.0$ and standard deviation $\sigma = 0.4$. Using $\Phi(1) = 0.8413$ and $\Phi(2) = 0.9772$, where $\Phi$ is the standard normal CDF, what is $P(4.6 < X \le 5.8)$?

- **(A)** $0.1359$
- **(B)** $0.8185$
- **(C)** $0.9545$
- **(D)** $0.1815$

> **Answer: (B).**
>
> Standardise both ends: $z = \frac{4.6-5.0}{0.4} = -1$ and $z = \frac{5.8-5.0}{0.4} = 2$. So $P(4.6 < X \le 5.8) = F(5.8) - F(4.6) = \Phi(2) - \Phi(-1)$. By the symmetry of the normal distribution $\Phi(-1) = 1 - \Phi(1) = 0.1587$, which gives $0.9772 - 0.1587 = 0.8185$. Taking $\Phi(-1)$ as $\Phi(1)$ gives $0.1359$; treating the interval as symmetric, $2\Phi(2) - 1$, gives $0.9545$; and $0.1815$ is the probability of falling outside the interval.

---

### Question 7

A bag contains 5 red and 3 green balls. Two balls are drawn at random without replacement. What is the probability that both are green?

- **(A)** $\tfrac{9}{64}$
- **(B)** $\tfrac{3}{32}$
- **(C)** $\tfrac{3}{28}$
- **(D)** $\tfrac{9}{56}$

> **Answer: (C).**
>
> By the product rule, $P(\text{both green}) = P(G_1)\,P(G_2\,|\,G_1) = \frac38 \cdot \frac27 = \frac{3}{28}$: after a green ball has been drawn, 2 of the 7 balls left are green. With replacement the draws would be independent, giving $\frac{9}{64}$. Reducing only the number of green balls gives $\frac38 \cdot \frac28 = \frac{3}{32}$, and reducing only the total gives $\frac38 \cdot \frac37 = \frac{9}{56}$.

---

### Question 8

The joint distribution of $X \in \{0, 1\}$ and $Y \in \{0, 1, 2\}$ is given in the table (rows: values of $X$; columns: values of $Y$). $$\begin{array}{c|ccc} & Y=0 & Y=1 & Y=2 \\ \hline X=0 & 0.10 & 0.20 & 0.10 \\ X=1 & 0.30 & 0.15 & 0.15 \end{array}$$ What is $P(Y=1\,|\,X=1)$?

- **(A)** $\tfrac{3}{20}$
- **(B)** $\tfrac{3}{7}$
- **(C)** $\tfrac{7}{20}$
- **(D)** $\tfrac{1}{4}$

> **Answer: (D).**
>
> By the sum rule $P(X=1) = 0.30 + 0.15 + 0.15 = 0.60$, and by the product rule $P(Y=1\,|\,X=1) = \frac{P(X=1, Y=1)}{P(X=1)} = \frac{0.15}{0.60} = \frac14$. The joint probability itself is $\frac{3}{20}$. Dividing it by $P(Y=1) = 0.35$ instead gives $\frac37$, which is $P(X=1\,|\,Y=1)$. And $P(Y=1) = \frac{7}{20}$ ignores the condition.

---

### Question 9

A part is manufactured by one of three machines $H_0, H_1, H_2$ with prior probabilities $P(H_0)=\tfrac{1}{2}$, $P(H_1)=\tfrac{3}{10}$, $P(H_2)=\tfrac{1}{5}$. The probability a part is defective ($E$) given its machine is $P(E\mid H_0)=\tfrac{1}{20}$, $P(E\mid H_1)=\tfrac{1}{10}$, $P(E\mid H_2)=\tfrac{3}{10}$. A part is found defective. What is $P(H_0 \mid E)$?

- **(A)** $\dfrac{1}{20}$
- **(B)** $\dfrac{6}{23}$
- **(C)** $\dfrac{1}{40}$
- **(D)** $\dfrac{5}{23}$

> **Answer: (D).**
>
> By Bayes' theorem, $P(H_0\mid E)=\dfrac{P(H_0)P(E\mid H_0)}{\sum_i P(H_i)P(E\mid H_i)}$. The joint terms are $\tfrac12\cdot\tfrac1{20}=\tfrac1{40}$, $\tfrac3{10}\cdot\tfrac1{10}=\tfrac3{100}$, $\tfrac15\cdot\tfrac3{10}=\tfrac3{50}$, summing to $\tfrac{23}{200}$. So $P(H_0\mid E)=\dfrac{1/40}{23/200}=\dfrac{5}{23}$. The value $1/40$ is the unnormalized joint term; $1/20$ is the raw likelihood; $6/23$ is the posterior for $H_1$, not $H_0$.

---

### Question 10

A spam filter uses naive Bayes to classify e-mails as spam or ham (legitimate e-mail), with two features: whether an e-mail contains the word $\textit{free}$, and whether it contains the word $\textit{meeting}$. The priors are $P(\text{spam}) = \tfrac25$ and $P(\text{ham}) = \tfrac35$, and the class-conditional probabilities are $P(\text{free}\,|\,\text{spam}) = \tfrac12$, $P(\text{free}\,|\,\text{ham}) = \tfrac{1}{10}$, $P(\text{no meeting}\,|\,\text{spam}) = \tfrac{9}{10}$ and $P(\text{no meeting}\,|\,\text{ham}) = \tfrac12$. An e-mail contains $\textit{free}$ but not $\textit{meeting}$. What is the posterior probability that it is spam?

- **(A)** $\tfrac{10}{13}$
- **(B)** $\tfrac{9}{10}$
- **(C)** $\tfrac{6}{7}$
- **(D)** $\tfrac{9}{50}$

> **Answer: (C).**
>
> Under the naive assumption (the features are conditionally independent given the class), each class's numerator is its prior times the product of the feature probabilities: spam $\frac25 \cdot \frac12 \cdot \frac{9}{10} = \frac{9}{50}$, ham $\frac35 \cdot \frac{1}{10} \cdot \frac12 = \frac{3}{100}$. Dividing by their sum, the evidence, gives $P(\text{spam}\,|\,\text{free, no meeting}) = \frac{9/50}{9/50 + 3/100} = \frac{18}{21} = \frac67$. Stopping at the numerator gives $\frac{9}{50}$. Leaving out the priors gives $\frac{9/20}{9/20 + 1/20} = \frac{9}{10}$. Using only the $\textit{free}$ feature gives $\frac{1/5}{1/5 + 3/50} = \frac{10}{13}$.

---
