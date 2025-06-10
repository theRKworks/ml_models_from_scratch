## Implemented Linear regression from scratch

In this project I have implemented linear regression from scratch by only using numpy and also included one test example in the code.

# 🧠 What is linear regression?

Linear regression is one of the most fundamental and basic machine learning model in which we try to draw a line which minimises the sum of squared distances form all the points and then use this line to predict the value of unknown points, it's very intuitive approach to solve regreesion tasks, let me elucidate with the indepth working of mathematics behind it📈.

- Consider line equation to be y = mx + c, where y is the matrix of independent variable, x is the matrix of dependent variables and c be the matrix of base constants.
- As discussed earlier. we'll try to minimise the sum of squared distances from all the points, therefore we'll calculate eucledian distance of all the points form the line by the formula (y_pred - y_i)**2
- In order to minimise it we'll use a technique called gradient descend, intuition behind using it is like coming down from slope of a hill to the bottom of the valley 
![Gradient Descend](https://pmc.ncbi.nlm.nih.gov/articles/PMC10426722/)
- so we'll use a thing called learning rate which will often be represented by alpha, the parameter decides how fast you are supposed to climb down the hill, climbing very fast might cause accidents, also it may be possible that you overshoot the minimum point and climbing very slow might cause sun to set and you to be food for wild animals. That's why choosing learning rate is important.
- Mostly learning rate is kept as 1e-3 or 0.001, but it's upto you what you want to keep. Also you can use various optimization techniques to improve your model accuracy.
- so finally we calculate somethind called a cost function or loss function which we supposed to minimise which is here the square of distances🧮.
- And then we'll use this equation to update the values of the parameters involved in the model:-
m = m - alpha*dm
c = c - alpha*dc, where
dm = 2*x*(mx + c - y_i)
dc = 2*(mx + c - y_i)
dm and dc are derivatives of the cost function with respect to m and c.