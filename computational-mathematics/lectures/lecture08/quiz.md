# Computational Mathematics -- Lecture 8 quiz

$\newcommand{\ii}{\mathrm{i}}\newcommand{\dd}{\mathrm{d}}\newcommand{\rr}{\mathbb{R}}\newcommand{\cc}{\mathbb{C}}$

### Question 1

For the dataset $\{1, 3, 7, 13\}$, compute the variance.

- **(A)** $21$
- **(B)** $28$
- **(C)** $57$
- **(D)** $4$

---

### Question 2

The data set $\{1, 3, 4, 5, 7\}$ is the whole population of interest. What is the z-score of the value 7?

- **(A)** $3.5$
- **(B)** $1.34$
- **(C)** $0.75$
- **(D)** $1.5$

---

### Question 3

Treating the paired data $x = (1, 2, 3, 4)$ and $y = (2, 4, 5, 9)$ as the whole population, what is the covariance $\mathrm{Cov}(x, y)$?

- **(A)** $\tfrac{61}{4}$
- **(B)** $\tfrac{11}{4}$
- **(C)** $\tfrac{11}{3}$
- **(D)** $11$

---

### Question 4

Two fair six-sided dice are rolled. What is the probability that the sum of the two numbers is 8 (to 3 decimal places)?

- **(A)** $0.139$
- **(B)** $0.091$
- **(C)** $0.143$
- **(D)** $0.083$

---

### Question 5

A random variable $X$ has probability density function $p(x) = c\,(4 - x)$ for $0 \le x \le 4$, and $p(x) = 0$ otherwise. What is the value of the constant $c$?

- **(A)** $\tfrac{1}{24}$
- **(B)** $\tfrac{1}{16}$
- **(C)** $\tfrac{1}{8}$
- **(D)** $\tfrac{1}{4}$

---

### Question 6

The sepal length $X$ (in cm) of a species of iris is modelled as normal with mean $\mu = 5.0$ and standard deviation $\sigma = 0.4$. Using $\Phi(1) = 0.8413$ and $\Phi(2) = 0.9772$, where $\Phi$ is the standard normal CDF, what is $P(4.6 < X \le 5.8)$?

- **(A)** $0.1359$
- **(B)** $0.8185$
- **(C)** $0.9545$
- **(D)** $0.1815$

---

### Question 7

A bag contains 5 red and 3 green balls. Two balls are drawn at random without replacement. What is the probability that both are green?

- **(A)** $\tfrac{9}{64}$
- **(B)** $\tfrac{3}{32}$
- **(C)** $\tfrac{3}{28}$
- **(D)** $\tfrac{9}{56}$

---

### Question 8

The joint distribution of $X \in \{0, 1\}$ and $Y \in \{0, 1, 2\}$ is given in the table (rows: values of $X$; columns: values of $Y$). $$\begin{array}{c|ccc} & Y=0 & Y=1 & Y=2 \\ \hline X=0 & 0.10 & 0.20 & 0.10 \\ X=1 & 0.30 & 0.15 & 0.15 \end{array}$$ What is $P(Y=1\,|\,X=1)$?

- **(A)** $\tfrac{3}{20}$
- **(B)** $\tfrac{3}{7}$
- **(C)** $\tfrac{7}{20}$
- **(D)** $\tfrac{1}{4}$

---

### Question 9

A part is manufactured by one of three machines $H_0, H_1, H_2$ with prior probabilities $P(H_0)=\tfrac{1}{2}$, $P(H_1)=\tfrac{3}{10}$, $P(H_2)=\tfrac{1}{5}$. The probability a part is defective ($E$) given its machine is $P(E\mid H_0)=\tfrac{1}{20}$, $P(E\mid H_1)=\tfrac{1}{10}$, $P(E\mid H_2)=\tfrac{3}{10}$. A part is found defective. What is $P(H_0 \mid E)$?

- **(A)** $\dfrac{1}{20}$
- **(B)** $\dfrac{6}{23}$
- **(C)** $\dfrac{1}{40}$
- **(D)** $\dfrac{5}{23}$

---

### Question 10

A spam filter uses naive Bayes to classify e-mails as spam or ham (legitimate e-mail), with two features: whether an e-mail contains the word $\textit{free}$, and whether it contains the word $\textit{meeting}$. The priors are $P(\text{spam}) = \tfrac25$ and $P(\text{ham}) = \tfrac35$, and the class-conditional probabilities are $P(\text{free}\,|\,\text{spam}) = \tfrac12$, $P(\text{free}\,|\,\text{ham}) = \tfrac{1}{10}$, $P(\text{no meeting}\,|\,\text{spam}) = \tfrac{9}{10}$ and $P(\text{no meeting}\,|\,\text{ham}) = \tfrac12$. An e-mail contains $\textit{free}$ but not $\textit{meeting}$. What is the posterior probability that it is spam?

- **(A)** $\tfrac{10}{13}$
- **(B)** $\tfrac{9}{10}$
- **(C)** $\tfrac{6}{7}$
- **(D)** $\tfrac{9}{50}$

---
