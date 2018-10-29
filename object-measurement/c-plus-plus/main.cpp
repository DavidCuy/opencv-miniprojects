#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <time.h>
#include <sstream>

using namespace std;
using namespace cv;

bool mouse_down, mouse_up; //Banderas para conocer el estado del mouse
Point corner1, corner2; //Puntos de esquina del recorte
Point aux;  //Se crea un punto auxiliar que sera por el que mouse se encuentre mientras este presionado
bool drawing;
int ref_h, ref_w, real_m = -1;
double measure;

static void mouse_callback(int event, int x, int y, int, void*);
float Calc_mk(int k, Mat Probabilidades);

int main()
{
    VideoCapture cap(1); // Se inicializa la camara default
    int heigth = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int with = cap.get(CV_CAP_PROP_FRAME_WIDTH);

    cout<<"Resolution: " << heigth << " x " << with <<endl
       << heigth * with << " Pixels" <<endl;

    if(!cap.isOpened())  // Verifica si puede capturar video
    {
        cout<<"No se puede iniciar la camara"<<endl;
        return -1;
    }
    namedWindow("Video");
    setMouseCallback("Video", mouse_callback);
    //namedWindow("Grises");

    for(;;)
    {
        Mat frame;
        bool getFrame = cap.read(frame);

        if(!getFrame)
        {
            cout<<"Error...\n No se pudo leer frame del video" <<endl;
            break;
        }

        if(drawing && mouse_down)
        {
            line(frame, corner1, aux, Scalar(255, 0, 0), 2);
        }
        imshow("Video", frame);

        int tecla = waitKey(30);

        if(tecla == 27) // Si presionas escape
        {
            break;
        }

        if(tecla == 13) // Si presionas enter
        {
            Mat g_frame, markers;
            int count;

            cvtColor(frame, g_frame, CV_BGR2GRAY);

            float mk;
            int bins = 256;

            float range[] { 0, 255 };
            const float* histRange = { range };

            Mat* segments;
            int fils = 1, cols = 1;

            int tam = fils*cols;
            segments = new Mat[tam];
            Mat edit = g_frame.clone();

            for(int i = 0; i < cols; i++)
            {
                for(int j = 0; j < fils; j++)
                {
                    segments[j + i * fils] = edit.rowRange(j * edit.rows/fils, (j + 1) * edit.rows/fils).colRange(i * edit.cols/cols, (i + 1) * edit.cols/cols);
                }

            }

            float T [tam];

            for(int i = 0; i < tam; i++)
            {
                Mat hist;
                int totalp = segments[i].rows * segments[i].cols;

                calcHist(&segments[i], 1, 0, Mat(), hist, 1, &bins, &histRange, true, false);

                hist /= totalp;

                Mat ojiv = hist.clone();
                Mat SigmaB = Mat(256, 1, CV_32F);
                float P1 = 0;

                for(int j = 1; j < ojiv.rows; j++)
                {
                    ojiv.at<float>(j) += ojiv.at<float>(j - 1);
                }

                float mG = 0;

                for(int j = 0; j < hist.rows; j++)
                {
                   mG += j * hist.at<float>(j);
                }

                float max = 0;

                for(int k = 0; k < hist.rows; k++)
                {
                    mk = Calc_mk(k, hist);
                    P1 = ojiv.at<float>(k);
                    if(P1 > 0 && P1 < 1)
                    {
                        SigmaB.at<float>(k) = pow(mG * P1 - mk, 2)/(P1 * (1 - P1));
                    }
                    else
                    {
                        SigmaB.at<float>(k) = 0;
                    }

                    if(max <= SigmaB.at<float>(k))
                    {
                        if(k > 0 && k < 255)
                        {
                            T[i] = k;
                            max = SigmaB.at<float>(k);
                        }
                    }
                }

                threshold(segments[i], segments[i], T[i], 255, THRESH_BINARY);
            }

            //Elimina espacios pequeños en la imagen
            erode(edit, edit, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
            dilate(edit, edit, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );

            //Rellena los pequeños agujeros en la imagen
            dilate( edit, edit, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
            erode(edit, edit, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );

            edit = ~edit; //sacamos el negaivo

            Mat th_a;
            adaptiveThreshold(edit, th_a, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 105, 0);

            vector<vector<Point> > c, contours;
            vector<Vec4i> hierarchy;
            findContours(th_a, c, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

            for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])
                if(contourArea(c[idx]) > 20) contours.push_back(c[idx]);

            count = contours.size();

            cout <<"Contornos: " << count <<endl;
            markers.create(edit.size(), CV_32SC1);

            for(int idx = 0; idx < contours.size(); idx++)
                drawContours(markers, contours, idx, Scalar::all(idx + 1), -1, 8);

            watershed(frame, markers);

            vector<Vec3b> colorTab;
            for(int i = 0; i < count; i++) {
                int b = theRNG().uniform(0, 255);
                int g = theRNG().uniform(0, 255);
                int r = theRNG().uniform(0, 255);

                colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
            }

            Mat object = Mat::zeros(frame.size(), CV_8UC3);

            for(int i = 0; i < markers.rows; i++)
            {
                for(int j = 0; j < markers.cols; j++)
                {
                    int index = markers.at<int>(i, j);

                    if(index > 0 && index <= count)
                    {
                        object.at<Vec3b>(i, j) = frame.at<Vec3b>(i, j);
                    }
                }
            }

            Mat object_m;
            cvtColor(object, object_m, CV_BGR2GRAY);

            Mat wshed(markers.size(), CV_8UC3);
            for(int i = 0; i < markers.rows; i++)
            {
                for(int j = 0; j < markers.cols; j++) {
                    int index = markers.at<int>(i, j);
                    if(index == -1)
                        wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
                    else if(index <= 0 || index > count)
                        wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                    else
                        wshed.at<Vec3b>(i, j) = colorTab[index - 1];
                }
            }

            Mat imgGray;
            cvtColor(edit, imgGray, CV_GRAY2BGR);
            wshed = wshed*0.5 + imgGray*0.5;

            Moments mnts = moments(edit);
            Point cent;

            cent.x = mnts.m10 / mnts.m00;
            cent.y = mnts.m01 / mnts.m00;

            circle(object, cent, 0, Scalar(0,0,255), 4);

            double mu20 = mnts.mu20 / mnts.m00;
            double mu02 = mnts.mu02 / mnts.m00;
            double mu11 = mnts.mu11 / mnts.m00;

            double disc = (4 * mu11 * mu11) + pow(mu20 - mu02, 2);
            double a = (mu20 + mu02) / 2;

            double lamda1 = a + sqrt(disc) / 2;
            double lamda2 = a - sqrt(disc) / 2;

            double exct = sqrt(1- lamda2/lamda1);
            double angle = 0.5 * atan((2 * mu11) / (mu20 - mu02));

            if(angle < 0)
                angle = 90 + (180 * angle / 3.14159);
            else
                angle = (180 * angle / 3.14159);

            vector<RotatedRect> minRect( contours.size() );
            vector<RotatedRect> minEllipse( contours.size() );

            for( int i = 0; i < contours.size(); i++ )
            {
                minRect[i] = minAreaRect( Mat(contours[i]) );
                if( contours[i].size() > 5 )
                {
                    minEllipse[i] = fitEllipse( Mat(contours[i]) );
                }

            }

            double line_mes[4];

            for( int i = 0; i< contours.size(); i++ )
            {
                Scalar colorR = Scalar(0, 255, 0);
                Scalar colorE = Scalar(255, 0, 0);
                // ellipse
                ellipse( frame, minEllipse[i], colorE, 2);
                // rotated rectangle
                Point2f rect_points[4];
                minRect[i].points( rect_points );
                for( int j = 0; j < 4; j++ )
                {
                      line( frame, rect_points[j], rect_points[(j+1)%4], colorR, 2);
                      line_mes[j] = pow(rect_points[j].x - rect_points[(j+1)%4].x, 2) + pow(rect_points[j].y - rect_points[(j+1)%4].y, 2);
                      line_mes[j] = sqrt(line_mes[j]);
                }
            }

            double heigth1;
            double width1;

            if(line_mes[0] < line_mes[1])
            {
                heigth1 = line_mes[1];
                width1 = line_mes[0];
            }
            else
            {
                heigth1 = line_mes[0];
                width1 = line_mes[1];
            }

            struct tm *tiempo;
            int dia;
            int mes;
            int anio;

            time_t date;
            time(&date);

            tiempo = localtime(&date);

            anio = tiempo->tm_year + 1900;
            mes = tiempo->tm_mon + 1;
            dia = tiempo->tm_mday;

            ostringstream os;
            os << dia << "-" << mes << "-" << anio << ".txt";
            string file_name = os.str();

            os.flush();

            ofstream archivo(file_name.c_str(), ios::app);

            if(real_m < 0)
            {
                cout << "Debe seleccionar una medida base"<<endl;
            }

            else
            {
                heigth1 = heigth1 * real_m / measure;
                width1 = width1 * real_m / measure;
                archivo <<"Angulo: " << angle <<"°;  Ancho: "<< width1 <<"cm;  Largo: "<<heigth1 <<"cm;  Excentricidad: " << exct <<endl;
            }

            archivo.close();

            ostringstream oswindow;
            oswindow << "Ancho: " << width1 << " cms   Largo: " << heigth1 << " cms.";
            string frame_text = oswindow.str();
            oswindow.flush();

            putText(frame, frame_text, Point(20,460), FONT_ITALIC, 0.8, CV_RGB(255, 0, 0), 2);

            namedWindow("Frame");

            imshow("Frame", frame);

            waitKey();
            count = 0;
            tecla = -1;
            ~frame;
            ~g_frame;
            ~markers;
            ~wshed;
            ~imgGray;
            ~edit;
            ~object;
            ~object_m;
            colorTab.empty();
            destroyWindow("Frame");

        }
    }

    return 0;
}

//Funcion callback del mouse
static void mouse_callback(int event, int x, int y, int, void*)
{
    if(event == EVENT_LBUTTONDOWN)  //Si el evento es tener presionado el mouse
    {
        drawing = true;
        mouse_down = true;  //La bandera de mouse preionado es verdadera
        corner1.x = x;  //Se haya la coordenada en x del click
        corner1.y = y;  //Se haya la coordenada en y del click

        cout<<"Inicio de recorte: ["<< corner1.x <<", "<< corner1.y <<"]"<<endl;
    }
    else if(event == EVENT_LBUTTONUP)   //Si el evento es haber levantado el mouse
    {
        drawing = false;
        mouse_up = true;    //La bandera de mouse preionado es verdadera

        corner2.x = x;
        corner2.y = y;

        cout<<"Fin de recorte: ["<< corner2.x <<", "<< corner2.y <<"]"<<endl;
    }

    //Evento para dibujar el rectangulo para señalar el corte
    if(mouse_down && !mouse_up) //Mientras tenga presionado el mouse
    {
        aux.x = x;
        aux.y = y;
    }

    else if(mouse_down && mouse_up) //Si se ya se realizaron los eventos de click y levantar
    {
        //Se añadiran los atributos al rectangulo original para crear la nueva imagen
        ref_w = abs(corner2.x - corner1.x); //Ancho
        ref_h = abs(corner2.y - corner1.y);    //Altura

        measure = sqrt((ref_h * ref_h) + (ref_w * ref_w));

        cout <<"[ " << ref_w <<", " << ref_h <<" , " << measure <<" ]"<<endl;

        cout <<"Medida = "<<measure<<endl<<endl<<endl;
        cout<<"Medida en cms de la linea: ";
        cin >> real_m;

        mouse_down = false;
        mouse_up = false;
    }
}

float Calc_mk(int k, Mat Probabilidades)
{
   float Mi_mk;
   Mi_mk = 0;
   for(int i = 0; i <= k; i++)
   {
       Mi_mk += i * Probabilidades.at<float>(i);
   }

   return Mi_mk;
}
