library(ggplot2)

nhanes_data <- readRDS("/Users/rushil/POCUS_B-Line_Prediction/create_image_metrics/NHANES/nhanes_fda_with_r.rds")
summary(nhanes_data)

ggplot(nhanes_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Age Distribution", x = "Age", y = "Frequency")
