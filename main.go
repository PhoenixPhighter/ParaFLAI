package main

import (
	"fmt"
	"log"
	// "os"

	// "github.com/go-gota/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	// irisCsv, err := os.Open("iris_headers.csv")
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// df := dataframe.ReadCSV(irisCsv)
	// head := df.Subset([]int{0, 3})

	// Filter by row values
	// versicolorOnly := df.Filter(dataframe.F{Colname: " Species", Comparator: "==", Comparando: "Iris-versicolor"})
	// fmt.Println(versicolorOnly)

	// Select specific columns
	// attrFiltered := df.Select([]string{"Petal length", "Sepal length"})
	// fmt.Println(attrFiltered)

	fmt.Println("Load our csv data")
	rawData, err := base.ParseCSVToInstances("iris_headers.csv", true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Initialize our KNN classifier")
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	fmt.Println("Perform a training-test split")
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	cls.Fit(trainData)

	fmt.Println("Make predictions")
	predictions, err := cls.Predict(testData)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(predictions)

	fmt.Println("Print our summary metrics")
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

}
