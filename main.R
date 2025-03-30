library(e1071)
library(caret)
library(class)
library(xgboost)
library(randomForest)
library(rpart)
library(ggplot2)

# wczytanie zbioru uczacego i walidacyjnego
df_train<- read.csv('jan_train.csv')
df_val <- read.csv('jan_test.csv')
# ograniczenie liczby wierszy dla zbioru uczacego
df_train_limited <- head(df_train, 100000)
# podzielenie zbioru uczacego na cechy x i etykiegy y
X_train <- df_train_limited[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]
y_train <- df_train_limited[, c("x", "y", "z", "Vx", "Vy", "Vz")]
########################################################## SVM #########################################################################
### Uczenie modelu SVM dla kazdej zmiennej ###
model_x <- svm(y_train$x ~ ., data = X_train)
model_y <- svm(y_train$y ~ ., data = X_train)
model_z <- svm(y_train$z ~ ., data = X_train)
model_Vx <- svm(y_train$Vx ~ ., data = X_train)
model_Vy <- svm(y_train$Vy ~ ., data = X_train)
model_Vz <- svm(y_train$Vz ~ ., data = X_train)
# Dodanie do zmiennej wartosci na podstawie ktorych bedziemy przewidywac ze zbioru walidacyjnego
X_val <- df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]
# Dokonanie predykcji
predictions_x <- predict(model_x, newdata = X_val)
predictions_y <- predict(model_y, newdata = X_val)
predictions_z <- predict(model_z, newdata = X_val)
predictions_Vx <- predict(model_Vx, newdata = X_val)
predictions_Vy <- predict(model_Vy, newdata = X_val)
predictions_Vz <- predict(model_Vz, newdata = X_val)
# Utworzenie ramki danych z predykcjami dla modelu SVM
predicted_df <- data.frame(
  x = predictions_x,
  y = predictions_y,
  z = predictions_z,
  Vx = predictions_Vx,
  Vy = predictions_Vy,
  Vz = predictions_Vz
)
# Wyświetlenie pierwszych wierszy utworzonej ramki danych
print(head(predicted_df))

########################################################## KNN #########################################################################
X_train <- df_train_limited[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]
y_train_x <- df_train_limited$x
y_train_y <- df_train_limited$y
y_train_z <- df_train_limited$z
y_train_Vx <- df_train_limited$Vx
y_train_Vy <- df_train_limited$Vy
y_train_Vz <- df_train_limited$Vz
# Ustaw liczbę sąsiadów
k <- 3
# Przewidywanie wartosci z wykorzystaniem modelu KNN
model_x <- knn(train = X_train, test = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")], cl = y_train_x, k = k)
model_y <- knn(train = X_train, test = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")], cl = y_train_y, k = k)
model_z <- knn(train = X_train, test = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")], cl = y_train_z, k = k)
model_Vx <- knn(train = X_train, test = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")], cl = y_train_Vx, k = k)
model_Vy <- knn(train = X_train, test = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")], cl = y_train_Vy, k = k)
model_Vz <- knn(train = X_train, test = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")], cl = y_train_Vz, k = k)
# Utworzenie ramki danych dla modelu KNN
predicted_df_knn <- data.frame(
  x = model_x,
  y = model_y,
  z = model_z,
  Vx = model_Vx,
  Vy = model_Vy,
  Vz = model_Vz
)
# Wyswietlenie pierwszych wierszy dla modelu KNN
print(head(predicted_df_knn))
########################################################## xgboost #########################################################################
X_train <- df_train_limited[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]
y_train_x <- df_train_limited$x
y_train_y <- df_train_limited$y
y_train_z <- df_train_limited$z
y_train_Vx <- df_train_limited$Vx
y_train_Vy <- df_train_limited$Vy
y_train_Vz <- df_train_limited$Vz
# Trenowanie modelu xgboost nie wykorzystuje listy i przekazuje do modelu, poniewaz otrzymawel bląd.
model_x <- xgboost(data = as.matrix(X_train), label = y_train_x,
                   objective = "reg:squarederror",
                   booster = "gbtree",
                   eval_metric = "rmse",
                   nrounds = 100,
                   max_depth = 3,
                   eta = 0.1)

model_y <- xgboost(data = as.matrix(X_train), label = y_train_y,
                   objective = "reg:squarederror",
                   booster = "gbtree",
                   eval_metric = "rmse",
                   nrounds = 100,
                   max_depth = 3,
                   eta = 0.1)

model_z <- xgboost(data = as.matrix(X_train), label = y_train_z,
                   objective = "reg:squarederror",
                   booster = "gbtree",
                   eval_metric = "rmse",
                   nrounds = 100,
                   max_depth = 3,
                   eta = 0.1)

model_Vx <- xgboost(data = as.matrix(X_train), label = y_train_Vx,
                    objective = "reg:squarederror",
                    booster = "gbtree",
                    eval_metric = "rmse",
                    nrounds = 100,
                    max_depth = 3,
                    eta = 0.1)

model_Vy <- xgboost(data = as.matrix(X_train), label = y_train_Vy,
                    objective = "reg:squarederror",
                    booster = "gbtree",
                    eval_metric = "rmse",
                    nrounds = 100,
                    max_depth = 3,
                    eta = 0.1)

model_Vz <- xgboost(data = as.matrix(X_train), label = y_train_Vz,
                    objective = "reg:squarederror",
                    booster = "gbtree",
                    eval_metric = "rmse",
                    nrounds = 100,
                    max_depth = 3,
                    eta = 0.1)
# Przewidywanie wartości dla wytrenowanego modelu
predictions_x <- predict(model_x, as.matrix(df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]))
predictions_y <- predict(model_y, as.matrix(df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]))
predictions_z <- predict(model_z, as.matrix(df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]))
predictions_Vx <- predict(model_Vx, as.matrix(df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]))
predictions_Vy <- predict(model_Vy, as.matrix(df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]))
predictions_Vz <- predict(model_Vz, as.matrix(df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")]))
# Utworzenie ramki danych z obliczonymi predykcjami
predicted_df_xgboost <- data.frame(
  x = predictions_x,
  y = predictions_y,
  z = predictions_z,
  Vx = predictions_Vx,
  Vy = predictions_Vy,
  Vz = predictions_Vz
)
# Wyświetl kilka pierwszych wierszy ramki danych
print(head(predicted_df_xgboost))
########################################################## RANDOM FOREST #########################################################################
#trenowanie modelu
model_x <- randomForest(x = X_train, y = y_train_x, ntree = 100, mtry = sqrt(ncol(X_train)))
model_y <- randomForest(x = X_train, y = y_train_y, ntree = 100, mtry = sqrt(ncol(X_train)))
model_z <- randomForest(x = X_train, y = y_train_z, ntree = 100, mtry = sqrt(ncol(X_train)))
model_Vx <- randomForest(x = X_train, y = y_train_Vx, ntree = 100, mtry = sqrt(ncol(X_train)))
model_Vy <- randomForest(x = X_train, y = y_train_Vy, ntree = 100, mtry = sqrt(ncol(X_train)))
model_Vz <- randomForest(x = X_train, y = y_train_Vz, ntree = 100, mtry = sqrt(ncol(X_train)))
#dokonanie predykcji
predictions_x <- predict(model_x, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_y <- predict(model_y, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_z <- predict(model_z, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_Vx <- predict(model_Vx, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_Vy <- predict(model_Vy, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_Vz <- predict(model_Vz, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
#utworzenie ramki danych
predicted_df_rf <- data.frame(
  x = predictions_x,
  y = predictions_y,
  z = predictions_z,
  Vx = predictions_Vx,
  Vy = predictions_Vy,
  Vz = predictions_Vz
)
#wyswietlenie pierwszych wierszy z ramki danych
print(head(predicted_df_rf))
########################################################## TREE #########################################################################
model_x <- rpart(formula = y_train_x ~ ., data = cbind(X_train, y_train_x))
model_y <- rpart(formula = y_train_y ~ ., data = cbind(X_train, y_train_y))
model_z <- rpart(formula = y_train_z ~ ., data = cbind(X_train, y_train_z))
model_Vx <- rpart(formula = y_train_Vx ~ ., data = cbind(X_train, y_train_Vx))
model_Vy <- rpart(formula = y_train_Vy ~ ., data = cbind(X_train, y_train_Vy))
model_Vz <- rpart(formula = y_train_Vz ~ ., data = cbind(X_train, y_train_Vz))
# Przewidywanie wartości
predictions_x <- predict(model_x, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_y <- predict(model_y, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_z <- predict(model_z, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_Vx <- predict(model_Vx, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_Vy <- predict(model_Vy, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
predictions_Vz <- predict(model_Vz, newdata = df_val[, c("x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim")])
# Utwórz ramkę danych z predykcjami
predicted_df_tree <- data.frame(
  x = predictions_x,
  y = predictions_y,
  z = predictions_z,
  Vx = predictions_Vx,
  Vy = predictions_Vy,
  Vz = predictions_Vz
)
# Wyświetlenie kilka pierwszych wierszy ramki danych
print(head(predicted_df_tree))
########################################################## RMSE - ROOT MEAN SQUERE ERROR #########################################################################
# wczytanie klucza odpowiedzi
zbior_oceniajacy <- read.csv('answer_key.csv')
# wyswietlenie nazw kolum w celu werfikacji czy pokrywaja sie z utworzonymi ramkami danych
colnames(zbior_oceniajacy)

true_values <- zbior_oceniajacy

# Definicja funkcji do obliczania RMSE
calculate_rmse <- function(predictions, true_values) {
  sqrt(mean((predictions - true_values)^2))
}
########################################################## RMSE TREE #########################################################################
# Oblicz RMSE dla każdej zmiennej ramki TREE
rmse_x <- calculate_rmse(predicted_df_tree$x, true_values$x)
rmse_y <- calculate_rmse(predicted_df_tree$y, true_values$y)
rmse_z <- calculate_rmse(predicted_df_tree$z, true_values$z)
rmse_Vx <- calculate_rmse(predicted_df_tree$Vx, true_values$Vx)
rmse_Vy <- calculate_rmse(predicted_df_tree$Vy, true_values$Vy)
rmse_Vz <- calculate_rmse(predicted_df_tree$Vz, true_values$Vz)
# Utwórz ramkę danych z wynikami RMSE
rmse_results <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RMSE_TREE = c(rmse_x, rmse_y, rmse_z, rmse_Vx, rmse_Vy, rmse_Vz)
)
# Wyświetl ramkę danych z wynikami
print(head(rmse_results))
########################################################## RMSE XGBOOST #########################################################################
rmse_x <- calculate_rmse(predicted_df_xgboost$x, true_values$x)
rmse_y <- calculate_rmse(predicted_df_xgboost$y, true_values$y)
rmse_z <- calculate_rmse(predicted_df_xgboost$z, true_values$z)
rmse_Vx <- calculate_rmse(predicted_df_xgboost$Vx, true_values$Vx)
rmse_Vy <- calculate_rmse(predicted_df_xgboost$Vy, true_values$Vy)
rmse_Vz <- calculate_rmse(predicted_df_xgboost$Vz, true_values$Vz)
# Utworzenie ramki danych z wynikami RMSE
rmse_results_xgboost <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RMSE_XGBOOST = c(rmse_x, rmse_y, rmse_z, rmse_Vx, rmse_Vy, rmse_Vz)
)
print(head(rmse_results_xgboost))
########################################################## RMSE RANDOM FOREST #########################################################################
rmse_rf_x <- calculate_rmse(predicted_df_rf$x, true_values$x)
rmse_rf_y <- calculate_rmse(predicted_df_rf$y, true_values$y)
rmse_rf_z <- calculate_rmse(predicted_df_rf$z, true_values$z)
rmse_rf_Vx <- calculate_rmse(predicted_df_rf$Vx, true_values$Vx)
rmse_rf_Vy <- calculate_rmse(predicted_df_rf$Vy, true_values$Vy)
rmse_rf_Vz <- calculate_rmse(predicted_df_rf$Vz, true_values$Vz)
# Utwórz ramkę danych z wynikami RMSE dla Random Forest
rmse_results_rf <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RMSE_RF = c(rmse_rf_x, rmse_rf_y, rmse_rf_z, rmse_rf_Vx, rmse_rf_Vy, rmse_rf_Vz)
)
# Wyświetl ramkę danych z wynikami
print(rmse_results_rf)
########################################################## RMSE KNN #########################################################################
# Konwersja kolumn na typ numeryczny
predicted_df_knn$x <- as.numeric(as.character(predicted_df_knn$x))
predicted_df_knn$y <- as.numeric(as.character(predicted_df_knn$y))
predicted_df_knn$z <- as.numeric(as.character(predicted_df_knn$z))
predicted_df_knn$Vx <- as.numeric(as.character(predicted_df_knn$Vx))
predicted_df_knn$Vy <- as.numeric(as.character(predicted_df_knn$Vy))
predicted_df_knn$Vz <- as.numeric(as.character(predicted_df_knn$Vz))
# Obliczenie RMSE
rmse_knn_x <- calculate_rmse(predicted_df_knn$x, true_values$x)
rmse_knn_y <- calculate_rmse(predicted_df_knn$y, true_values$y)
rmse_knn_z <- calculate_rmse(predicted_df_knn$z, true_values$z)
rmse_knn_Vx <- calculate_rmse(predicted_df_knn$Vx, true_values$Vx)
rmse_knn_Vy <- calculate_rmse(predicted_df_knn$Vy, true_values$Vy)
rmse_knn_Vz <- calculate_rmse(predicted_df_knn$Vz, true_values$Vz)
# Utworzenie ramki danych z wynikami RMSE dla KNN
rmse_results_knn <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RMSE_KNN = c(rmse_knn_x, rmse_knn_y, rmse_knn_z, rmse_knn_Vx, rmse_knn_Vy, rmse_knn_Vz)
)
# Wyświetlenie ramki danych z wynikami
print(rmse_results_knn)
########################################################## RMSE SVM #########################################################################
rmse_svm_x <- calculate_rmse(predicted_df$x, true_values$x)
rmse_svm_y <- calculate_rmse(predicted_df$y, true_values$y)
rmse_svm_z <- calculate_rmse(predicted_df$z, true_values$z)
rmse_svm_Vx <- calculate_rmse(predicted_df$Vx, true_values$Vx)
rmse_svm_Vy <- calculate_rmse(predicted_df$Vy, true_values$Vy)
rmse_svm_Vz <- calculate_rmse(predicted_df$Vz, true_values$Vz)
#Utworzenie ramki danych z wynikami
rmse_results_svm <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RMSE_SVM = c(rmse_svm_x, rmse_svm_y, rmse_svm_z, rmse_svm_Vx, rmse_svm_Vy, rmse_svm_Vz)
)
#wyswietl otrzymane wyniki
print(rmse_results_svm)
########################################################## WSPOLCZYNNIK DETERMINANCJI #########################################################################
# zdefiniowanie funkcji sluzaczej do obliczanie wspolczynnika determinancji
calculate_rsquared <- function(predictions, true_values) {
  1 - (sum((predictions - true_values)^2) / sum((true_values - mean(true_values))^2))
}
########################################################## Wspolczynnik determinancji dla TREE #########################################################################
rsquared_tree_x <- calculate_rsquared(predicted_df_tree$x, true_values$x)
rsquared_tree_y <- calculate_rsquared(predicted_df_tree$y, true_values$y)
rsquared_tree_z <- calculate_rsquared(predicted_df_tree$z, true_values$z)
rsquared_tree_Vx <- calculate_rsquared(predicted_df_tree$Vx, true_values$Vx)
rsquared_tree_Vy <- calculate_rsquared(predicted_df_tree$Vy, true_values$Vy)
rsquared_tree_Vz <- calculate_rsquared(predicted_df_tree$Vz, true_values$Vz)
#utworz ramke danych z otrzymanych wynikow
rsquared_results_tree <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RSQUARED_TREE = c(rsquared_tree_x, rsquared_tree_y, rsquared_tree_z, rsquared_tree_Vx, rsquared_tree_Vy, rsquared_tree_Vz)
)
#wyswietlenie otrzymanego wyniku
print("R dla drzewa decyzyjnego:")
print(rsquared_results_tree)
########################################################## Wspolczynnik determinancji dla XGBOOST #########################################################################
rsquared_xgboost_x <- calculate_rsquared(predicted_df_xgboost$x, true_values$x)
rsquared_xgboost_y <- calculate_rsquared(predicted_df_xgboost$y, true_values$y)
rsquared_xgboost_z <- calculate_rsquared(predicted_df_xgboost$z, true_values$z)
rsquared_xgboost_Vx <- calculate_rsquared(predicted_df_xgboost$Vx, true_values$Vx)
rsquared_xgboost_Vy <- calculate_rsquared(predicted_df_xgboost$Vy, true_values$Vy)
rsquared_xgboost_Vz <- calculate_rsquared(predicted_df_xgboost$Vz, true_values$Vz)
# Utwórz ramkę danych z wynikami dla XGBoost
rsquared_results_xgboost <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RSQUARED_XGBOOST = c(rsquared_xgboost_x, rsquared_xgboost_y, rsquared_xgboost_z, rsquared_xgboost_Vx, rsquared_xgboost_Vy, rsquared_xgboost_Vz)
)
# Wyświetl ramkę danych z wynikami dla XGBoost
print("R dla XGBoost:")
print(rsquared_results_xgboost)
########################################################## Wspolczynnik determinancji dla RANDOM FOREST #########################################################################
rsquared_rf_x <- calculate_rsquared(predicted_df_rf$x, true_values$x)
rsquared_rf_y <- calculate_rsquared(predicted_df_rf$y, true_values$y)
rsquared_rf_z <- calculate_rsquared(predicted_df_rf$z, true_values$z)
rsquared_rf_Vx <- calculate_rsquared(predicted_df_rf$Vx, true_values$Vx)
rsquared_rf_Vy <- calculate_rsquared(predicted_df_rf$Vy, true_values$Vy)
rsquared_rf_Vz <- calculate_rsquared(predicted_df_rf$Vz, true_values$Vz)
# Utwórz ramkę danych z wynikami dla Random Forest
rsquared_results_rf <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RSQUARED_RF = c(rsquared_rf_x, rsquared_rf_y, rsquared_rf_z, rsquared_rf_Vx, rsquared_rf_Vy, rsquared_rf_Vz)
)
# Wyświetl ramkę danych z wynikami R-squared dla Random Forest
print("R dla Random Forest:")
print(rsquared_results_rf)
########################################################## Wspolczynnik determinancji dla KNN #########################################################################
rsquared_knn_x <- calculate_rsquared(predicted_df_knn$x, true_values$x)
rsquared_knn_y <- calculate_rsquared(predicted_df_knn$y, true_values$y)
rsquared_knn_z <- calculate_rsquared(predicted_df_knn$z, true_values$z)
rsquared_knn_Vx <- calculate_rsquared(predicted_df_knn$Vx, true_values$Vx)
rsquared_knn_Vy <- calculate_rsquared(predicted_df_knn$Vy, true_values$Vy)
rsquared_knn_Vz <- calculate_rsquared(predicted_df_knn$Vz, true_values$Vz)
# Utwórz ramkę danych z wynikami dla KNN
rsquared_results_knn <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RSQUARED_KNN = c(rsquared_knn_x, rsquared_knn_y, rsquared_knn_z, rsquared_knn_Vx, rsquared_knn_Vy, rsquared_knn_Vz)
)
# Wyświetl ramkę danych z wynikami dla KNN
print("R dla KNN:")
print(rsquared_results_knn)
########################################################## Wspolczynnik determinancji dla SVM #########################################################################
rsquared_svm_x <- calculate_rsquared(predicted_df$x, true_values$x)
rsquared_svm_y <- calculate_rsquared(predicted_df$y, true_values$y)
rsquared_svm_z <- calculate_rsquared(predicted_df$z, true_values$z)
rsquared_svm_Vx <- calculate_rsquared(predicted_df$Vx, true_values$Vx)
rsquared_svm_Vy <- calculate_rsquared(predicted_df$Vy, true_values$Vy)
rsquared_svm_Vz <- calculate_rsquared(predicted_df$Vz, true_values$Vz)
# Utwórz ramkę danych z wynikami dla SVM
rsquared_results_svm <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RSQUARED_SVM = c(rsquared_svm_x, rsquared_svm_y, rsquared_svm_z, rsquared_svm_Vx, rsquared_svm_Vy, rsquared_svm_Vz)
)
# Wyświetl ramkę danych z wynikami dla SVM
print("R dla SVM:")
print(rsquared_results_svm)
########################################################## OBLICZANIE MAE - MEAN ABSOLUTE ERROR #########################################################################
# definicja funkcji obliczajacej mae
calculate_mae <- function(predictions, true_values) {
  mean(abs(predictions - true_values))
}
########################################################## MAE dla TREE #########################################################################
mae_tree_x <- calculate_mae(predicted_df_tree$x, true_values$x)
mae_tree_y <- calculate_mae(predicted_df_tree$y, true_values$y)
mae_tree_z <- calculate_mae(predicted_df_tree$z, true_values$z)
mae_tree_Vx <- calculate_mae(predicted_df_tree$Vx, true_values$Vx)
mae_tree_Vy <- calculate_mae(predicted_df_tree$Vy, true_values$Vy)
mae_tree_Vz <- calculate_mae(predicted_df_tree$Vz, true_values$Vz)
########################################################## MAE dla XGBOOST #########################################################################
mae_xgboost_x <- calculate_mae(predicted_df_xgboost$x, true_values$x)
mae_xgboost_y <- calculate_mae(predicted_df_xgboost$y, true_values$y)
mae_xgboost_z <- calculate_mae(predicted_df_xgboost$z, true_values$z)
mae_xgboost_Vx <- calculate_mae(predicted_df_xgboost$Vx, true_values$Vx)
mae_xgboost_Vy <- calculate_mae(predicted_df_xgboost$Vy, true_values$Vy)
mae_xgboost_Vz <- calculate_mae(predicted_df_xgboost$Vz, true_values$Vz)
########################################################## MAE dla RANDOM FOREST #########################################################################
mae_rf_x <- calculate_mae(predicted_df_rf$x, true_values$x)
mae_rf_y <- calculate_mae(predicted_df_rf$y, true_values$y)
mae_rf_z <- calculate_mae(predicted_df_rf$z, true_values$z)
mae_rf_Vx <- calculate_mae(predicted_df_rf$Vx, true_values$Vx)
mae_rf_Vy <- calculate_mae(predicted_df_rf$Vy, true_values$Vy)
mae_rf_Vz <- calculate_mae(predicted_df_rf$Vz, true_values$Vz)
########################################################## MAE dla KNN #########################################################################
mae_knn_x <- calculate_mae(predicted_df_knn$x, true_values$x)
mae_knn_y <- calculate_mae(predicted_df_knn$y, true_values$y)
mae_knn_z <- calculate_mae(predicted_df_knn$z, true_values$z)
mae_knn_Vx <- calculate_mae(predicted_df_knn$Vx, true_values$Vx)
mae_knn_Vy <- calculate_mae(predicted_df_knn$Vy, true_values$Vy)
mae_knn_Vz <- calculate_mae(predicted_df_knn$Vz, true_values$Vz)
########################################################## MAE dla SVM #########################################################################
mae_svm_x <- calculate_mae(predicted_df$x, true_values$x)
mae_svm_y <- calculate_mae(predicted_df$y, true_values$y)
mae_svm_z <- calculate_mae(predicted_df$z, true_values$z)
mae_svm_Vx <- calculate_mae(predicted_df$Vx, true_values$Vx)
mae_svm_Vy <- calculate_mae(predicted_df$Vy, true_values$Vy)
mae_svm_Vz <- calculate_mae(predicted_df$Vz, true_values$Vz)
#utworzenie ramki danych z otrzymanymi wartosciami MAE
mae_results <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  MAE_TREE = c(mae_tree_x, mae_tree_y, mae_tree_z, mae_tree_Vx, mae_tree_Vy, mae_tree_Vz),
  MAE_XGBOOST = c(mae_xgboost_x, mae_xgboost_y, mae_xgboost_z, mae_xgboost_Vx, mae_xgboost_Vy, mae_xgboost_Vz),
  MAE_RF = c(mae_rf_x, mae_rf_y, mae_rf_z, mae_rf_Vx, mae_rf_Vy, mae_rf_Vz),
  MAE_KNN = c(mae_knn_x, mae_knn_y, mae_knn_z, mae_knn_Vx, mae_knn_Vy, mae_knn_Vz),
  MAE_SVM = c(mae_svm_x, mae_svm_y, mae_svm_z, mae_svm_Vx, mae_svm_Vy, mae_svm_Vz)
)
# Wyświetl ramkę danych z wynikami MAE
print(mae_results)
#utworzenie wynikow zbiorczych dla wspolczynnika determinancji
r_results <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  R_SVM = c(rsquared_svm_x, rsquared_svm_y, rsquared_svm_z, rsquared_svm_Vx, rsquared_svm_Vy, rsquared_svm_Vz),
  R_KNN = c(rsquared_knn_x, rsquared_knn_y, rsquared_knn_z, rsquared_knn_Vx, rsquared_knn_Vy, rsquared_knn_Vz),
  R_RF = c(rsquared_rf_x, rsquared_rf_y, rsquared_rf_z, rsquared_rf_Vx, rsquared_rf_Vy, rsquared_rf_Vz),
  R_XGBOOST = c(rsquared_xgboost_x, rsquared_xgboost_y, rsquared_xgboost_z, rsquared_xgboost_Vx, rsquared_xgboost_Vy, rsquared_xgboost_Vz),
  R_TREE = c(rsquared_tree_x, rsquared_tree_y, rsquared_tree_z, rsquared_tree_Vx, rsquared_tree_Vy, rsquared_tree_Vz)
)
print(r_results)
# utworzenie zbiorczych wynikow dla RMSE
rmse_zbiorcze <- data.frame(
  Variable = c("x", "y", "z", "Vx", "Vy", "Vz"),
  RMSE_TREE = c(rmse_x, rmse_y, rmse_z, rmse_Vx, rmse_Vy, rmse_Vz),
  RMSE_KNN = c(rmse_knn_x, rmse_knn_y, rmse_knn_z, rmse_knn_Vx, rmse_knn_Vy, rmse_knn_Vz),
  RMSE_RF = c(rmse_rf_x, rmse_rf_y, rmse_rf_z, rmse_rf_Vx, rmse_rf_Vy, rmse_rf_Vz),
  RMSE_XGBOOST = c(rmse_x, rmse_y, rmse_z, rmse_Vx, rmse_Vy, rmse_Vz),
  RMSE_SVM = c(rmse_svm_x, rmse_svm_y, rmse_svm_z, rmse_svm_Vx, rmse_svm_Vy, rmse_svm_Vz)
)
########################################################## ANALIZA I WIZUALIZACJA #########################################################################
#przygotowanie danych
true_values_long <- reshape2::melt(true_values)
# Wykres pudełkowy dla x, y, z
ggplot(subset(true_values_long, variable %in% c("x", "y", "z")), aes(x = variable, y = value)) +
  geom_boxplot() +
  stat_summary(fun=mean, geom="point", shape=20, size=3, color="red") +  # Oznaczanie średniej
  stat_summary(fun=median, geom="point", shape=4, size=3, color="blue") +  # Oznaczanie mediany
  labs(title = "Boxplot dla x, y, z",
       x = "Zmienna",
       y = "Wartość") +
  theme_minimal()
# Wykres pudełkowy dla Vx, Vy, Vz
ggplot(subset(true_values_long, variable %in% c("Vx", "Vy", "Vz")), aes(x = variable, y = value)) +
  geom_boxplot() +
  stat_summary(fun=mean, geom="point", shape=20, size=3, color="red") +  # Oznaczanie średniej
  stat_summary(fun=median, geom="point", shape=4, size=3, color="blue") +  # Oznaczanie mediany
  labs(title = "Boxplot dla Vx, Vy, Vz",
       x = "Zmienna",
       y = "Wartość") +
  theme_minimal()
#################################### Wykresy MAE dla poszczególnych zmiennych #########################
mae_long <- reshape2::melt(mae_results, id.vars = "Variable")
# Zmienne do pierwszego wykresu
variables_xyz <- c("x", "y", "z")
# Wykres dla x, y, z
ggplot(mae_long[mae_long$Variable %in% variables_xyz, ], aes(x = variable, y = value, color = variable, group = variable)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Średnie oraz MAE dla zmiennych x, y, z",
       x = "Zmienna",
       y = "Wartość") +
  theme_minimal() +
  theme(legend.position = "top",
        axis.text.x = element_blank()) +
  scale_color_manual(values = c("MAE_TREE" = "blue", "MAE_XGBOOST" = "green", "MAE_RF" = "purple", "MAE_KNN" = "orange", "MAE_SVM" = "brown")) +
  facet_wrap(~Variable)
# Zmienne do drugiego wykresu
variables_vxvyvz <- c("Vx", "Vy", "Vz")
# Wykres dla Vx, Vy, Vz
ggplot(mae_long[mae_long$Variable %in% variables_vxvyvz, ], aes(x = variable, y = value, color = variable, group = variable)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "Średnie oraz MAE dla zmiennych Vx, Vy, Vz",
       x = "Zmienna",
       y = "Wartość") +
  theme_minimal() +
  theme(legend.position = "top",
        axis.text.x = element_blank()) +
  scale_color_manual(values = c("MAE_TREE" = "blue", "MAE_XGBOOST" = "green", "MAE_RF" = "purple", "MAE_KNN" = "orange", "MAE_SVM" = "brown")) +
  facet_wrap(~Variable)
#################################### Wykresy determinancji dla poszczególnych zmiennych #########################
r_results_long <- tidyr::gather(r_results, Model, R_Squared, -Variable)

# Narysuj wykres liniowy
ggplot(r_results_long, aes(x = Variable, y = R_Squared, color = Model, group = Model)) +
  geom_point() +
  labs(title = "R dla różnych modeli",
       x = "Zmienna",
       y = "R-squared") +
  theme_minimal()
#################################### Wykresy RMSE dla poszczególnych zmiennych i modeli #########################
# Długie dane do analizy
rmse_zbiorcze_long <- reshape2::melt(rmse_zbiorcze, id.vars = "Variable")
# Zmienne do pierwszego wykresu
variables_xyz <- c("x", "y", "z")
# Wykres dla x, y, z
ggplot(rmse_zbiorcze_long[rmse_zbiorcze_long$Variable %in% variables_xyz, ], aes(x = Variable, y = value, color = variable, group = variable)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "RMSE dla zmiennych x, y, z",
       x = "Zmienna",
       y = "Wartość") +
  theme_minimal() +
  theme(legend.position = "top",
        axis.text.x = element_blank()) +
  scale_color_manual(values = c("RMSE_TREE" = "blue", "RMSE_KNN" = "green", "RMSE_RF" = "purple", "RMSE_XGBOOST" = "orange", "RMSE_SVM" = "brown")) +
  facet_wrap(~Variable)
# Zmienne do drugiego wykresu
variables_vxvyvz <- c("Vx", "Vy", "Vz")
# Wykres dla Vx, Vy, Vz
ggplot(rmse_zbiorcze_long[rmse_zbiorcze_long$Variable %in% variables_vxvyvz, ], aes(x = Variable, y = value, color = variable, group = variable)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(title = "RMSE dla zmiennych Vx, Vy, Vz",
       x = "Zmienna",
       y = "Wartość") +
  theme_minimal() +
  theme(legend.position = "top",
        axis.text.x = element_blank()) +
  scale_color_manual(values = c("RMSE_TREE" = "blue", "RMSE_KNN" = "green", "RMSE_RF" = "purple", "RMSE_XGBOOST" = "orange", "RMSE_SVM" = "brown")) +
  facet_wrap(~Variable)

########################################################## PODSUMOWANIE ########################################################
# Badany przeze mnie zbior pobrany ze strony kaggle ( link - https://www.kaggle.com/datasets/idawoodjee/predict-the-positions-and-speeds-of-600-satellites/data?select=jan_test.csv)
# podejmuje ciekawe zagadnienia przewidywania realnego polożenia satelity w przestrzeni euklidesowej oraz wektor szybkosci w tej przestrzeni ( odpowiednio wspolrzedne x,y,z oraz Vx,Vy,Vz)
# na podstawie symulowanych danych ( zmienne poprzedzone przedrostkiem sim_).
#
# Zestaw danych został podzielony przez autora na 3 pliki csv:
# answer_key.csv - plik sluzacy do oceny predykcji
# jan_test.csv - plik sluzacy do walidacji
# jan_train.csv - plik sluzacy do trenowania modelu
#
# (ze względu na ograniczoną moc obliczeniową mojego urzadzenia oraz bardzo duzy rozmiar plikow ktore posiadaja ponad milion rekordow)
# zmuszony zostalem do ograniczenia tych zbiorow w linijce 13 kodu gdzie wybieram okreslona liczbe wierszy.
# Zdecydowalem się jednak skorzystać z tej sposobnosci oraz opracowac wyniki dla różnych rozmiarów zbiorów uczących, co swietnie ilustruje jak wazne jest
# dobranie odpowiedniego zbioru uczacego i jak rzutuje na jego ocene.
#
# wytrenowalem 30 różnych modeli, ponieważ mając na uwadzę, że predyukuje polożenie w przestrzenie euklidesowej każda zmienna zależy od pozostałych stąd wytrenowanie
# osobnego modelu dla kazdej zmiennej. Daje to ciekawe rezultaty, ponieważ podczas oceny modeli poprzez współczynnik determinancji zauważamy, że poszczególny model dla kazdej zmiennej
# bardzo często pozyskuje różne wartości wspolczynnikow. Utworzylem równiez zbiorcze ramki danych z 3 typami oceny modelu - RMSE, MSE, Wspolczynnik determinancji
#
# W ramach przygotowania danych poczatkowo dodalem petle warunkowa sprawdzajaca czy dany wiersz wchodzacy do mo modelu nie zawiera wybrakowanych wartosci
# jednak nie bylo takich sytuacji co spowodowalo, ze zdecydowalem sie usunac ten element z mojego kodu, poniewaz znacząco zwiększał złożoność obliczeniową programu
#
# Na wykresach oceny modeli nie nanosze wartosci sredniej ani mediany, poniewaz jak mozemy zauwazyc z wykresach pudelkowych dla okreslonych zmiennych
# wartosci te wystepuje w zerze / sa bliskie zeru dlatego do jako punkt odniesienia dla RMSE ora MAE nalezy brac rząd wartości wystepujący na wykresie boxplot.
#
# Naturalnie jako ostateczną ocenę wykorzystam wyniki dla najwiekszego zbioru uczenia - wykresy_10000 nie robiłbym tego w momencie gdy istnialoby zagrozenie
# overfittingu ale na podstawie moich wynikow nie mamy doczynienia z takim zagrozeniem
#
# zauwazamy ze najmniejszy sredni blad bezwgledny w predykcji dla wspolrzednych x y z popelnil algorytm xgboost natomiast najgorzej wypadl svm
# dla okreslenia wektorow predkosci rowniez xgboost wypada najlepiej z kolei model knn wypada najgorzej pozostale modele uzyskaly wyniki zblizone do xgboosta
# co ciekawe wspolczynnik determinancji dla zmiennych x y z choc wszystkie modele uzyskaly podobny wynik poza knn to na czele nie stoi xgboost ktory mial najmniejsze
# srednie bledy tylko model random forest z kolei dla predkosci najwieksza wartosc wspolczynnika osiaga xgboost
#
# w przypadku rmse dla wspolrzednych xgboost ponownie poradzil sobie najlepiej uzysukujac najmnieszy wynik  dla predkosci wszystkie modele poza knn uzyskaly zblizone wyniki
#
#