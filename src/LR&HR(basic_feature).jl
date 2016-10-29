using DataFrames
# HuberLR from https://github.com/bicycle1885/HuberLR.jl
include("HuberLR.jl")
train = readtable("E:\\ORIE4741\\project\\1021\\train.csv")
#store = readtable("E:\\ORIE4741\\project\\1021\\store.csv")

# Only train with open stores
train_open = train[train[:Open] .== 1, :]

# Function to convert categorical data to 0-1 Arrays
function convert_to_binary(data)
  uniqueData = unique(data);
  uniqueData = sort(uniqueData[:])
  finalData = zeros(length(data), length(uniqueData));
  for i = 1:length(uniqueData)
    tempInd = find(data .== uniqueData[i]);
    finalData[tempInd, i] = 1;
  end
  return finalData;
end

# Feature Transformation
dayOfWeek = convert_to_binary(convert(Array,train_open[:DayOfWeek]));
stHoliday = convert_to_binary(convert(Array,train_open[:StateHoliday]));
scHoliday = convert_to_binary(convert(Array,train_open[:SchoolHoliday]));

sales = convert(Array,train_open[:Sales]);
customer = convert(Array, train_open[:Customers]);
promo = convert(Array, train_open[:Promo]);
scHoliday = scHoliday[:,2];

# Fit with Linear Regression
X = [ones(size(train_open,1),1) dayOfWeek promo stHoliday scHoliday];
y = sales;
w = X\y;

# Fit with Huber Regression
y1 = convert(Array{Float64}, y)
lr = HuberLR.LinearRegression(1);
HuberLR.fit!(lr, X[:,2:size(train_open,2)],y1)

# Predict
w1 = w;
w2 = lr.w;

# Fuction to compute RMSPE, only consider those store have sale >0
function computeRMSPE(X,y,w)
  a = find(y .>0);
  X1 = X[a,:];
  y1 = y[a,:];
  a = (y1 - X1*w)./y1;
  b = sum(a.^2)/length(y);
  return sqrt(b);
end

X = [ones(size(train_open,1),1) dayOfWeek promo stHoliday scHoliday];

print(computeRMSPE(X,y,w1))
print(computeRMSPE(X,y,w2))

# Read test set
test = readtable("E:\\ORIE4741\\project\\Rossmann\\test.csv")

# Feature Transformation
dayOfWeek = convert_to_binary(convert(Array,test[:DayOfWeek]));
scHoliday = convert_to_binary(convert(Array,test[:SchoolHoliday]));
promo = convert(Array, test[:Promo]);
scHoliday = scHoliday[:,1];

# Test set only has 2 unique StateHoliday values, therefore has to process individually
uniqueSet = unique(train_open[:StateHoliday])
uniqueSet = sort(uniqueSet[:])
stHoliday = zeros(size(test,1), length(uniqueSet));
for i = 1:length(uniqueSet)
    tempInd = find(convert(Array,test[:StateHoliday]) .== uniqueSet[i]);
    stHoliday[tempInd, i] = 1;
end

# Predict Test Set
X_test = [ones(size(test,1),1) dayOfWeek promo stHoliday scHoliday];
y1 = X_test*w1;
y2 = X_test*w2;
open = convert(Array, train_open[:Open],1);
temp = find(open .==0);
y1[temp] = 0;
y2[temp] = 0

# Write out the result
output1 = DataFrame(Id = 1:length(y2),Sales = y1);
output2 = DataFrame(Id = 1:length(y2),Sales = y2);
writetable("E:\\ORIE4741\\project\\1028\\output1028(linear).csv", output1);
writetable("E:\\ORIE4741\\project\\1028\\output1028(huber).csv", output2);
