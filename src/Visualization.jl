using DataFrames, Gadfly
train = readtable("E:\\ORIE4741\\project\\1021\\train.csv")
store = readtable("E:\\ORIE4741\\project\\1021\\store.csv")
train_open = train[train[:Open] .== 1, :]
combined = join(train_open, store, on = :Store, kind = :inner)
ticks = [1, 2,  3, 4, 5, 6, 7]

## DayOfWeek
plot(train_open, x=:DayOfWeek, y=:Sales, Geom.violin, Guide.xticks(ticks=ticks), Coord.cartesian(xmin=0, xmax=8, ymin=0, ymax=50000))
## draw(PNG("E:\\ORIE4741\\project\\1021\\DayofWeek.png",6inch, 3inch), plot(train_open, x=:DayOfWeek, y=:Sales, Geom.violin))

## StoreType
plot(combined, x=:StoreType, y=:Sales, Geom.violin)
## Assortment
plot(combined, x=:Assortment, y=:Sales, Geom.violin)
