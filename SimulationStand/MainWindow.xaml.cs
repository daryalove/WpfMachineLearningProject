using ComputerVisionLib.Models;
using ComputerVisionLib.Services;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace SimulationStand
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        Thread backgroundThread;
        MLContext mlContext;
        List<ModelOutput> ListOfMessage;

        public MainWindow()
        {
            //InitializeComponent() — это вызов метода LoadComponent() класса System.Windows.Application.
            //Метод LoadComponent() извлекает код BAML(скомпилированный XAML) из сборки 
            //и использует его для построения пользовательского интерфейса.
            InitializeComponent();

            //В результате инициализации mlContext создается среда ML.NET, которая может использоваться 
            //всеми объектами в рамках процесса создания модели. По существу он аналогичен классу DbContext в Entity Framework.
            mlContext = new MLContext();
            ListOfMessage = new List<ModelOutput>();

            backgroundThread = new Thread(new ThreadStart(StartLoadImages));
            backgroundThread.IsBackground = true;

        }

        private void Btn_Click_Exit(object sender, RoutedEventArgs e)
        {
            Application.Current.Shutdown();
        }
        private void Btn_Click_Show(object sender, RoutedEventArgs e)
        {
            backgroundThread.Start();
        }

        private void OutputPrediction(ModelOutput prediction)
        {

            string imageName = System.IO.Path.GetFileName(prediction.ImagePath);
            prediction.ImagePath = imageName;
            ListOfMessage.Add(prediction);

            Dispatcher.BeginInvoke(new ThreadStart(delegate
            {
                listV.ItemsSource = ListOfMessage;
                listV.DataContext = this;
            }));

        }

        public void StartLoadImages()
        {
            DeepLearningService.LoadImages(ref mlContext);
            //DeepLearningService.LTP = 1;
            var prediction = DeepLearningService.ClassifySingleImage(ref mlContext);
            OutputPrediction(prediction);
        }

    }
}
