using ComputerVisionLib.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace ComputerVisionLib.Services
{
    public class DeepLearningService
    {
        private static string projectDirectory = System.IO.Path.GetFullPath(System.IO.Path.Combine(AppContext.BaseDirectory, "../../../"));
        private static string workspaceRelativePath = System.IO.Path.Combine(projectDirectory, "workspace");
        private static string assetsRelativePath = System.IO.Path.Combine(projectDirectory, "assets");
        private static string path = Directory.GetCurrentDirectory() + "\\model.zip";
        private static IDataView testSet;
        private static ITransformer trainedModel;
        private static IDataView shuffledData;
        //public static int LTP = 0;
        //private static string PathToImage = "7001-21";

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
    searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                //if(LTP == 1)
                //{
                //    if((Path.GetFileName(file) != "7001-21.jpg") )
                //    {
                //        continue;
                //    }
                //}
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }

        public static void LoadImages(ref MLContext mlContext)
        {
            // Получение списка изображений, используемых для обучения.
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

            // Загрузка избражений в IDataView
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

            // Данные загружаются в том порядке, в котором они были считаны из каталогов. 
            //Чтобы сбалансировать данные, перемешайте их в случайном порядке с помощью метода ShuffleRows.
            shuffledData = mlContext.Data.ShuffleRows(imageData);

            //Модели машинного обучения ожидают входные данные в числовом формате. 
            //Поэтому перед обучением необходимо выполнить некоторую предварительную обработку данных.(Создание класса)
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelAsKey")
            .Append(mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: assetsRelativePath,
                inputColumnName: "ImagePath"));

            //Используем метод Fit, чтобы применить данные к preprocessingPipelineEstimatorChain,
            //а затем метод Transform, который возвращает IDataView, содержащий предварительно обработанные данные.
            IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);


            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            testSet = validationTestSplit.TestSet;

            if (File.Exists(path))
            {
                //Define DataViewSchema for data preparation pipeline and trained model
                DataViewSchema modelSchema;

                // Load trained model
                trainedModel = mlContext.Model.Load(path, out modelSchema);
            }
            else
            {
                // Defining the learning pipeline
                var classifierOptions = new ImageClassificationTrainer.Options()
                {
                    FeatureColumnName = "Image",
                    LabelColumnName = "LabelAsKey",
                    ValidationSet = validationSet,
                    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    TestOnTrainSet = false,
                    ReuseTrainSetBottleneckCachedValues = true,
                    ReuseValidationSetBottleneckCachedValues = true,
                    WorkspacePath = workspaceRelativePath,

                };

                var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                trainedModel = trainingPipeline.Fit(trainSet);

                mlContext.Model.Save(trainedModel, trainSet.Schema, path);

            }
        }

        public static ModelOutput ClassifySingleImage(ref MLContext mlContext)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(testSet, reuseRowObject: true).First();
            // Классификация изображения.
            ModelOutput prediction = predictionEngine.Predict(image);

            return prediction;
        }
    }
}
