using System;
using System.Collections.Generic;
using System.Text;

namespace ComputerVisionLib.Models
{
    public class ImageData
    {
        //полный путь, по которому хранится изображение
        public string ImagePath { get; set; }

        //прогнозируемое значение
        public string Label { get; set; }
    }

    public class ModelInput
    {
        //данные изображений для обучения
        public byte[] Image { get; set; }

        //численное представление Label
        public UInt32 LabelAsKey { get; set; }

        //полный путь, по которому хранится изображение
        public string ImagePath { get; set; }

        //прогнозируемое значение
        public string Label { get; set; }
    }

    public class ModelOutput
    {
        //полный путь, по которому хранится изображение
        public string ImagePath { get; set; }

        //прогнозируемое значение
        public string Label { get; set; }

        //значение, спрогнозированное моделью
        public string PredictedLabel { get; set; }
    }
}
