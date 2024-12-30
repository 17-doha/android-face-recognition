package com.irhammuch.android.facerecognition;

import org.tensorflow.lite.Interpreter;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.File;
import java.io.IOException;

public class Embedding {

    private Interpreter tfliteInterpreter;

    // Constructor that loads the model
    public Embedding(String modelPath) throws IOException {
        // Load the TFLite model
        tfliteInterpreter = new Interpreter(loadModelFile(modelPath));
    }

    // Method to load the TFLite model from the file system
    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        FileInputStream fileInputStream = new FileInputStream(modelPath);
        FileChannel fileChannel = fileInputStream.getChannel();
        long fileSize = fileChannel.size();
        MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
        fileInputStream.close();
        return modelBuffer;
    }

    // Method to preprocess and get embeddings for a bitmap image
    public float[] getEmbeddings(Bitmap bitmap) throws IOException {
        // Resize and normalize the image to 160x160
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true);

        // Convert the bitmap to a ByteBuffer
        ByteBuffer byteBuffer = preprocessImage(resizedBitmap);

        // Prepare an output buffer for the embeddings
        float[][] output = new float[1][192]; // Assuming the output is a 192-dimensional vector

        // Run inference with the model to get the embedding
        tfliteInterpreter.run(byteBuffer, output);

        return output[0];
    }

    // Method to preprocess the image to a ByteBuffer
    private ByteBuffer preprocessImage(Bitmap bitmap) {
        // Create a ByteBuffer to store the image data
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(160 * 160 * 3 * 4); // 160x160x3 (RGB) * 4 bytes per float
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[160 * 160];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Normalize the image data to [0, 1] range and store in the ByteBuffer
        for (int i = 0; i < 160; i++) {
            for (int j = 0; j < 160; j++) {
                int pixel = pixels[i * 160 + j];
                byteBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // Red channel
                byteBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // Green channel
                byteBuffer.putFloat((pixel & 0xFF) / 255.0f);         // Blue channel
            }
        }

        byteBuffer.rewind();
        return byteBuffer;
    }

    // Method to convert a Bitmap to a byte array for saving or further use
    public byte[] convertBitmapToByteArray(Bitmap bitmap) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        return outputStream.toByteArray();
    }

    // Close the interpreter when done
    public void close() {
        tfliteInterpreter.close();
    }
}

