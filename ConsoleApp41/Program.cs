using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EmguCVHumanFaceDetectDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            const string ImageFilePath = @"..\..\Images\Demo1.png";
            const string ClassifierFilePath = @"..\..\Classifiers\haarcascade_frontalface_alt.xml";

            CvInvoke.UseOpenCL = CvInvoke.HaveOpenCLCompatibleGpuDevice;

            using (var face = new CascadeClassifier(ClassifierFilePath))
            {
                using (var img = new Image<Bgr, byte>(ImageFilePath))
                {
                    using (var img2 = new Image<Gray, byte>(img.ToBitmap()))
                    {
                        CvInvoke.CvtColor(img, img2, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
                        CvInvoke.EqualizeHist(img2, img2);

                        var size = new Size(100, 100);
                        var facesResult = face.DetectMultiScale(img2, 1.1, 10, size);

                        int count = 0;
                        var b = img.ToBitmap();
                        foreach (var item in facesResult)
                        {
                            count++;
                            var bmpOut = new Bitmap(item.Width, item.Height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                            var g = Graphics.FromImage(bmpOut);
                            g.DrawImage(b, new Rectangle(0, 0, item.Width, item.Height), new Rectangle(item.X, item.Y, item.Width, item.Height), GraphicsUnit.Pixel);
                            g.Dispose();
                            bmpOut.Save($@"..\..\Images\{count}.png", System.Drawing.Imaging.ImageFormat.Png);
                            bmpOut.Dispose();
                        }
                    }//end using img2
                }//end using img
            }//end using face
        }
    }
}
