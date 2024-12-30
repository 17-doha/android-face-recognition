package com.irhammuch.android.facerecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.Log;
import android.util.Pair;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import androidx.camera.core.ExperimentalGetImage;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_CODE = 1001;
    private static final String CAMERA_PERMISSION = Manifest.permission.CAMERA;
    private PreviewView previewView;
    private CameraSelector cameraSelector;
    private ProcessCameraProvider cameraProvider;
    private int lensFacing = CameraSelector.LENS_FACING_BACK;
    private Preview previewUseCase;
    private ImageAnalysis analysisUseCase;
    private GraphicOverlay graphicOverlay;
    private ImageView previewImg;
    private TextView detectionTextView;

    private final HashMap<String, SimilarityClassifier.Recognition> registered = new HashMap<>(); //saved Faces
    private Interpreter tfLite;
    private boolean flipX = false;
    private boolean start = true;
    private float[][] embeddings;

    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    private static final int INPUT_SIZE = 160;
    private static final int OUTPUT_SIZE=42;
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private Bitmap capturedBitmap;
    private String detectedClassName = null;

    @Override
    @ExperimentalGetImage
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);



        setContentView(R.layout.activity_main);
        Button saveButton = findViewById(R.id.save_button);
        previewView = findViewById(R.id.previewView);
        previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
        graphicOverlay = findViewById(R.id.graphic_overlay);
        previewImg = findViewById(R.id.preview_img);
        detectionTextView = findViewById(R.id.detection_text);




        ImageButton switchCamBtn = findViewById(R.id.switch_camera);
        switchCamBtn.setOnClickListener((view -> switchCamera()));

        loadModel();

        saveButton.setOnClickListener(v -> {
            if (detectedClassName != null) {
                saveToCSV(detectedClassName); // Save the detected class name to CSV
                Toast.makeText(this, "Saved to CSV: " + detectedClassName, Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "No class detected to save.", Toast.LENGTH_SHORT).show();
            }
        });
    }



    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            capturedBitmap = (Bitmap) extras.get("data");

            if (capturedBitmap != null) {
                classifyImage(capturedBitmap);
            }
        }
    }

    private void classifyImage(Bitmap bitmap) {
        // Resize bitmap if necessary
        String className = null;
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 320, 320, true);
        className = classifyFace(resizedBitmap);

        // Debugging log for result
        Log.d(TAG, "Classify result: " + className);

        // Write className to attendance.csv
        saveToCSV(className);

        // Display the result
        Toast.makeText(this, "Detected Class: " + className, Toast.LENGTH_SHORT).show();
    }

    private void saveToCSV(String name) {
        // Get the current date and time
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
        String currentTime = sdf.format(new Date());

        // Create a file in the Downloads directory
        String fileName = "Attendance.csv";
        File downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        File file = new File(downloadsDir, fileName);

        try (FileWriter writer = new FileWriter(file, true)) {
            // Add headers if the file is empty
            if (!file.exists() || file.length() == 0) {
                writer.append("Name, Date, Time\n");
            }

            // Append the name, date, and time to the file
            writer.append(name)
                    .append(", ")
                    .append(currentTime)
                    .append("\n");

            // Notify the user of the save
            Toast.makeText(this, name + " saved to Downloads", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error saving file", Toast.LENGTH_SHORT).show();
        }
    }



    @Override
    @ExperimentalGetImage
    protected void onResume() {
        super.onResume();
        startCamera();
    }

    /** Permissions Handler */
    private void getPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{CAMERA_PERMISSION}, PERMISSION_CODE);
    }

    @Override
    @ExperimentalGetImage
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, int[] grantResults) {
        for (int r : grantResults) {
            if (r == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
                return;
            }
        }

        if (requestCode == PERMISSION_CODE) {
            setupCamera();
        }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    /** Setup camera & use cases */
    @ExperimentalGetImage
    private void startCamera() {
        if(ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED) {
            setupCamera();
        } else {
            getPermissions();
        }
    }
    @ExperimentalGetImage
    private void setupCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraSelector = new CameraSelector.Builder().requireLensFacing(lensFacing).build();

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindAllCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "cameraProviderFuture.addListener Error", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }
    @ExperimentalGetImage
    private void bindAllCameraUseCases() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            bindPreviewUseCase();
            bindAnalysisUseCase();
        }
    }

    private void bindPreviewUseCase() {
        if (cameraProvider == null) {
            return;
        }

        if (previewUseCase != null) {
            cameraProvider.unbind(previewUseCase);
        }

        Preview.Builder builder = new Preview.Builder();
        builder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        builder.setTargetRotation(getRotation());

        previewUseCase = builder.build();
        previewUseCase.setSurfaceProvider(previewView.getSurfaceProvider());

        try {
            cameraProvider
                    .bindToLifecycle(this, cameraSelector, previewUseCase);
        } catch (Exception e) {
            Log.e(TAG, "Error when bind preview", e);
        }
    }
    @ExperimentalGetImage
    private void bindAnalysisUseCase() {
        if (cameraProvider == null) {
            return;
        }

        if (analysisUseCase != null) {
            cameraProvider.unbind(analysisUseCase);
        }

        Executor cameraExecutor = Executors.newSingleThreadExecutor();

        ImageAnalysis.Builder builder = new ImageAnalysis.Builder();
        builder.setTargetAspectRatio(AspectRatio.RATIO_4_3);
        builder.setTargetRotation(getRotation());

        analysisUseCase = builder.build();
        analysisUseCase.setAnalyzer(cameraExecutor, this::analyze);

        try {
            cameraProvider
                    .bindToLifecycle(this, cameraSelector, analysisUseCase);
        } catch (Exception e) {
            Log.e(TAG, "Error when bind analysis", e);
        }
    }

    protected int getRotation() throws NullPointerException {
        return previewView.getDisplay().getRotation();
    }

    @ExperimentalGetImage
    private void switchCamera() {
        if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            lensFacing = CameraSelector.LENS_FACING_FRONT;
            flipX = true;
        } else {
            lensFacing = CameraSelector.LENS_FACING_BACK;
            flipX = false;
        }

        if(cameraProvider != null) cameraProvider.unbindAll();
        startCamera();
    }




    @ExperimentalGetImage
    private void analyze(@NonNull ImageProxy imageProxy) {
        // Get the image object from the ImageProxy
        Image image = imageProxy.getImage();  // This is how you access the Image object

        if (image == null) {
            // Handle error or return if the image is not valid
            Log.e(TAG, "Image is null");
            imageProxy.close();  // Don't forget to close ImageProxy
            return;
        }

        // Use getImageInfo().getRotationDegrees() to retrieve rotation degrees
        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();

        InputImage inputImage = InputImage.fromMediaImage(
                image, rotationDegrees  // Use the rotationDegrees from imageInfo
        );

        // Process the image for face detection
        FaceDetector faceDetector = FaceDetection.getClient();
        faceDetector.process(inputImage)
                .addOnSuccessListener(faces -> onFaceDetected(faces, inputImage))
                .addOnFailureListener(e -> Log.e(TAG, "Face detection failure", e))
                .addOnCompleteListener(task -> imageProxy.close()); // Close imageProxy after processing
    }



    private void onFaceDetected(List<Face> faces, InputImage inputImage) {
        Rect boundingBox = null;
        String className = null;
        float scaleX = (float) previewView.getWidth() / (float) inputImage.getHeight();
        float scaleY = (float) previewView.getHeight() / (float) inputImage.getWidth();

        if (faces.size() > 0) {
            detectionTextView.setText(R.string.face_detected);

            // Get first face detected
            Face face = faces.get(0);

            // Get bounding box of face
            boundingBox = face.getBoundingBox();

            // Convert image to bitmap & crop based on bounding box
            Bitmap bitmap = mediaImgToBmp(inputImage.getMediaImage(), inputImage.getRotationDegrees(), boundingBox);

            // Resize the cropped bitmap to match the model input size (e.g., 160x160)
            Bitmap resizedBitmap = getResizedBitmap(bitmap);

            // Pass the face to the model for classification
            className = classifyFace(resizedBitmap);
            detectedClassName = className; // Store the detected class name
            detectionTextView.setText(className != null ? className : "Unknown");

        } else {
            detectedClassName = null; // Reset if no face is detected
            detectionTextView.setText(R.string.no_face_detected);
        }

        graphicOverlay.draw(boundingBox, scaleX, scaleY, className);
    }

    private String classifyFace(final Bitmap bitmap) {
        // Resize the bitmap to the expected size (160x160)
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true);

        // Ensure the bitmap size matches the expected input size (160x160)
        if (resizedBitmap.getWidth() != INPUT_SIZE || resizedBitmap.getHeight() != INPUT_SIZE) {
            Log.e(TAG, "Bitmap dimensions are incorrect. Expected " + INPUT_SIZE + "x" + INPUT_SIZE + ", but got "
                    + resizedBitmap.getWidth() + "x" + resizedBitmap.getHeight());
            return "Unknown";
        }

        // Set image to preview
        previewImg.setImageBitmap(resizedBitmap);

        // Create ByteBuffer to store normalized image
        ByteBuffer imgData = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4); // 160x160x3
        imgData.order(ByteOrder.nativeOrder());

        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

        // Get pixel values from Bitmap to normalize
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

        imgData.rewind();

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }

        // Pass the data to the model
        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        embeddings = new float[1][OUTPUT_SIZE]; // Output of the model will be stored here
        outputMap.put(0, embeddings);

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap); // Run model

        // Log the output embeddings
        Log.d(TAG, "Model output embeddings: " + Arrays.toString(embeddings[0]));

        // Now compare with the CSV embeddings
        EmbeddingMatcher matcher = new EmbeddingMatcher(this); // Pass the context
        try {
            String bestMatchClass = matcher.getBestMatch(embeddings[0], "embeddings (3).csv");
            return bestMatchClass;
        } catch (IOException e) {
            Log.e(TAG, "Error while loading or comparing embeddings", e);
            return "Unknown";
        }
    }






    /** Bitmap Converter */
    private Bitmap mediaImgToBmp(Image image, int rotation, Rect boundingBox) {
        //Convert media image to Bitmap
        Bitmap frame_bmp = toBitmap(image);

        //Adjust orientation of Face
        Bitmap frame_bmp1 = rotateBitmap(frame_bmp, rotation, flipX);

        //Crop out bounding box from whole Bitmap(image)
        float padding = 0.0f;
        RectF adjustedBoundingBox = new RectF(
                boundingBox.left - padding,
                boundingBox.top - padding,
                boundingBox.right + padding,
                boundingBox.bottom + padding);
        Bitmap cropped_face = getCropBitmapByCPU(frame_bmp1, adjustedBoundingBox);

        //Resize bitmap to 112,112
        return getResizedBitmap(cropped_face);
    }

    private Bitmap getResizedBitmap(Bitmap bm) {
        int targetWidth = 320;  // Change to 320
        int targetHeight = 320; // Change to 320

        // Create a new bitmap with the target dimensions
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bm, targetWidth, targetHeight, true);

        // Ensure the image is in RGB format
        Bitmap rgbBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, false);

        // Recycle the original bitmap to free memory if different
        if (bm != resizedBitmap) {
            bm.recycle();
        }

        return rgbBitmap;
    }




    private static Bitmap getCropBitmapByCPU(Bitmap source, RectF cropRectF) {
        Bitmap resultBitmap = Bitmap.createBitmap((int) cropRectF.width(),
                (int) cropRectF.height(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(resultBitmap);

        // draw background
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setColor(Color.WHITE);
        canvas.drawRect(//from  w w  w. ja v  a  2s. c  om
                new RectF(0, 0, cropRectF.width(), cropRectF.height()),
                paint);

        Matrix matrix = new Matrix();
        matrix.postTranslate(-cropRectF.left, -cropRectF.top);

        canvas.drawBitmap(source, matrix, paint);

        if (source != null && !source.isRecycled()) {
            source.recycle();
        }

        return resultBitmap;
    }

    private static Bitmap rotateBitmap(
            Bitmap bitmap, int rotationDegrees, boolean flipX) {
        Matrix matrix = new Matrix();

        // Rotate the image back to straight.
        matrix.postRotate(rotationDegrees);

        // Mirror the image along the X or Y axis.
        matrix.postScale(flipX ? -1.0f : 1.0f, 1.0f);
        Bitmap rotatedBitmap =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // Recycle the old bitmap if it has changed.
        if (rotatedBitmap != bitmap) {
            bitmap.recycle();
        }
        return rotatedBitmap;
    }


    private static byte[] YUV_420_888toNV21(Image image) {

        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width*height;
        int uvSize = width*height/4;

        byte[] nv21 = new byte[ySize + uvSize*2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // V

        int rowStride = image.getPlanes()[0].getRowStride();
        assert(image.getPlanes()[0].getPixelStride() == 1);

        int pos = 0;

        if (rowStride == width) { // likely
            yBuffer.get(nv21, 0, ySize);
            pos += ySize;
        }
        else {
            long yBufferPos = -rowStride; // not an actual position
            for (; pos<ySize; pos+=width) {
                yBufferPos += rowStride;
                yBuffer.position((int) yBufferPos);
                yBuffer.get(nv21, pos, width);
            }
        }

        rowStride = image.getPlanes()[2].getRowStride();
        int pixelStride = image.getPlanes()[2].getPixelStride();

        assert(rowStride == image.getPlanes()[1].getRowStride());
        assert(pixelStride == image.getPlanes()[1].getPixelStride());

        if (pixelStride == 2 && rowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            // maybe V an U planes overlap as per NV21, which means vBuffer[1] is alias of uBuffer[0]
            byte savePixel = vBuffer.get(1);
            try {
                vBuffer.put(1, (byte)~savePixel);
                if (uBuffer.get(0) == (byte)~savePixel) {
                    vBuffer.put(1, savePixel);
                    vBuffer.position(0);
                    uBuffer.position(0);
                    vBuffer.get(nv21, ySize, 1);
                    uBuffer.get(nv21, ySize + 1, uBuffer.remaining());

                    return nv21; // shortcut
                }
            }
            catch (ReadOnlyBufferException ex) {
                // unfortunately, we cannot check if vBuffer and uBuffer overlap
            }

            // unfortunately, the check failed. We must save U and V pixel by pixel
            vBuffer.put(1, savePixel);
        }

        // other optimizations could check if (pixelStride == 1) or (pixelStride == 2),
        // but performance gain would be less significant

        for (int row=0; row<height/2; row++) {
            for (int col=0; col<width/2; col++) {
                int vuPos = col*pixelStride + row*rowStride;
                nv21[pos++] = vBuffer.get(vuPos);
                nv21[pos++] = uBuffer.get(vuPos);
            }
        }

        return nv21;
    }



    private Bitmap toBitmap(Image image) {

        byte[] nv21=YUV_420_888toNV21(image);


        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    /** Model loader */
    @SuppressWarnings("deprecation")
    private void loadModel() {
        try {
            //model name
            String modelFile = "model2.tflite";
            tfLite = new Interpreter(loadModelFile(MainActivity.this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public class EmbeddingMatcher {

        private final Context context;

        public EmbeddingMatcher(Context context) {
            this.context = context;
        }

        // Load embeddings from the CSV file
        public List<Embedding> loadEmbeddings(String assetFileName) throws IOException {
            List<Embedding> embeddingsList = new ArrayList<>();
            AssetManager assetManager = context.getAssets();

            // Open the file as InputStream
            InputStream inputStream = assetManager.open(assetFileName);
            BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));

            String line;
            // Skip header if any
            br.readLine();

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");

                // Assuming the last column is the class and the rest are embedding values
                String className = values[values.length - 1];
                float[] embeddingValues = new float[values.length - 1];
                for (int i = 0; i < embeddingValues.length; i++) {
                    embeddingValues[i] = Float.parseFloat(values[i]);
                }

                embeddingsList.add(new Embedding(className, embeddingValues));
            }

            br.close();
            return embeddingsList;
        }

        public class Embedding {
            String className;
            float[] values;

            public Embedding(String className, float[] values) {
                this.className = className;
                this.values = values;
            }
        }
        public String getBestMatch(float[] modelEmbedding, String csvFile) throws IOException {
            List<Embedding> embeddingsList = loadEmbeddings(csvFile);

            String bestClass = "Unknown";
            float bestDistance = Float.MAX_VALUE;

            for (Embedding embedding : embeddingsList) {
                // Calculate the Euclidean distance between the model's embedding and the current embedding
                float distance = calculateCosineSimilarity(modelEmbedding, embedding.values);

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClass = embedding.className;
                }
            }

            return bestClass;
        }

        public float calculateEuclideanDistance(float[] modelEmbedding, float[] csvEmbedding) {
            float sumSquaredDifferences = 0;

            // Calculate sum of squared differences
            for (int i = 0; i < modelEmbedding.length; i++) {
                float difference = modelEmbedding[i] - csvEmbedding[i];
                sumSquaredDifferences += difference * difference;
            }

            // Return square root of sum
            return (float) Math.sqrt(sumSquaredDifferences);
        }

        public float calculateL1Distance(float[] modelEmbedding, float[] csvEmbedding) {
            float distance = 0;

            // Calculate L1 (Manhattan) distance: sum of absolute differences
            for (int i = 0; i < modelEmbedding.length; i++) {
                distance += Math.abs(modelEmbedding[i] - csvEmbedding[i]);
            }

            return distance;
        }

        // Calculate Euclidean distance
        public float calculateCosineSimilarity(float[] modelEmbedding, float[] csvEmbedding) {
            float dotProduct = 0;
            float modelMagnitude = 0;
            float csvMagnitude = 0;

            // Ensure both embeddings have the same length
            for (int i = 0; i < modelEmbedding.length; i++) {
                dotProduct += modelEmbedding[i] * csvEmbedding[i];
                modelMagnitude += modelEmbedding[i] * modelEmbedding[i];
                csvMagnitude += csvEmbedding[i] * csvEmbedding[i];
            }

            // Calculate magnitudes
            modelMagnitude = (float) Math.sqrt(modelMagnitude);
            csvMagnitude = (float) Math.sqrt(csvMagnitude);

            // Handle division by zero
            if (modelMagnitude == 0 || csvMagnitude == 0) {
                return 0; // Assume no similarity if one vector is zero
            }

            // Cosine similarity formula
            return dotProduct / (modelMagnitude * csvMagnitude);
        }


    }



}


