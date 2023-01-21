import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class NeuralNetwork{
    public static void main(String[] args) {

        long programStartTime = System.currentTimeMillis();

        List<List<Double>> trainDatatList = new ArrayList<>();
        List<List<Double>> testDatatList = new ArrayList<>();

        System.out.println("==========Start ReadInput==========");
        long readInputStartTime = System.currentTimeMillis();
        readInput(args, trainDatatList, testDatatList);
        System.out.println("readInputTime: " + (System.currentTimeMillis() - readInputStartTime));
        System.out.println("==========Finish ReadInput==========");

        int layerSize = 3;
        int inputNodeSize = 5;
        int hiddenNodeSize = 8;
        int outputNodeSize = 1;

        Random random = new Random();
        double wMin = -1.0;
        double wMax = 1.0;
        double bMin = 0.0;
        double bMax = 0.0;

        int epoch = 1000;
        int miniBatch = 1;
        double learnRate = 0.03;
        double lossMin = 0.001;
        String activationType = "sigmoid";
        String lossType = "mse";

        System.out.println("==========Start BuildNetwork==========");
        long buildNetworkStartTime = System.currentTimeMillis();
        List<Layer> layerList = new ArrayList<>();
        buildNetwork(layerList, layerSize, inputNodeSize, hiddenNodeSize, outputNodeSize, random, wMin, wMax, bMin, bMax);
        System.out.println("buildNetworkTime: " + (System.currentTimeMillis() - buildNetworkStartTime));
        System.out.println("==========Finish BuildNetwork==========");

        printLayerList(layerList, 0, -1);

        System.out.println("==========Start Train==========");
        long trainStartTime = System.currentTimeMillis();
        train(layerList, trainDatatList, epoch, miniBatch, learnRate, lossMin, activationType, lossType);
        System.out.println("trainTime: " + (System.currentTimeMillis() - trainStartTime));
        System.out.println("==========Finish Train==========");

        System.out.println("==========Start Predict==========");
        long predictStartTime = System.currentTimeMillis();
        List<Double> predictList = predict(layerList, testDatatList, activationType, inputNodeSize);
        System.out.println("predictTime: " + (System.currentTimeMillis() - predictStartTime));
        System.out.println("==========Finish Predict==========");

        System.out.println("==========Start WritePredict==========");
        String predictFileName = "test_predictions.csv";
        writePredict(predictList, predictFileName);
        System.out.println("writePredictTime: " + (System.currentTimeMillis() - programStartTime));
        System.out.println("==========Finish WritePredict==========");

        printLayerList(layerList, 0, -1);

        System.out.println("programTotalTime: " + (System.currentTimeMillis() - programStartTime));

    }

    public static void buildNetwork(List<Layer> layerList, int layerSize, int inputNodeSize, int hiddenNodeSize, int outputNodeSize, Random random, double wMin, double wMax, double bMin, double bMax) {
        for(int a = 0; a < layerSize; a++){
            Layer layer = null;
            if(a == 0){
                layer = new Layer("input", inputNodeSize, null, null, null, null, new double[inputNodeSize]);
            }
            else if(a == layerSize - 1){
                int preNodeSize = layerList.get(layerList.size() - 1).nodeSize;
                layer = new Layer("output", outputNodeSize, initialize2DArray(preNodeSize, outputNodeSize, random, wMin, wMax), initialize1DArray(outputNodeSize, random, bMin, bMax), new double[outputNodeSize][preNodeSize], new double[outputNodeSize], new double[outputNodeSize]);
            }
            else{
                int preNodeSize = layerList.get(layerList.size() - 1).nodeSize;
                layer = new Layer("hidden", hiddenNodeSize, initialize2DArray(preNodeSize, hiddenNodeSize, random, wMin, wMax), initialize1DArray(hiddenNodeSize, random, bMin, bMax), new double[hiddenNodeSize][preNodeSize], new double[hiddenNodeSize], new double[hiddenNodeSize]);
            }
            layerList.add(layer);
        }
    }

    public static void forward(List<Layer> layerList, String activationType) {
        for(int a = 1; a < layerList.size(); a++){
            Layer curLayer = layerList.get(a);
            Layer preLayer = layerList.get(a - 1);

            for(int c = 0; c < curLayer.nodeSize; c++){
                double tempZ = 0.0;
                for(int d = 0; d < preLayer.nodeSize; d++){
                    tempZ += curLayer.w[c][d] * preLayer.h[d];
                }
                tempZ += curLayer.b[c];
                curLayer.z[c] = tempZ;
            }
            for(int c = 0; c < curLayer.nodeSize; c++){
                curLayer.h[c] = calcActivation(activationType, curLayer.z[c]);
            }
        }
    }

    public static void backword(List<Layer> layerList, String activationType, String lossType, List<Double> labelList) {
        for(int a = layerList.size() - 1; a >= 1; a--){
            Layer curLayer = layerList.get(a);
            Layer nextLayer = null;
            if(a + 1 < layerList.size()){
                nextLayer = layerList.get(a + 1);
            }

            if("output".equals(curLayer.layerType)){
                for(int c = 0; c < curLayer.nodeSize; c++){
                    for(int d = 0; d < curLayer.partialDerivative[c].length; d++){
                        curLayer.partialDerivative[c][d] = calcLossDerivative(lossType, labelList.get(c), curLayer.h[c]) * calcActivationDerivative(activationType, curLayer.h[c]);
                    }
                }
            }
            else{
                for(int c = 0; c < curLayer.nodeSize; c++){
                    double tempSum = 0.0;
                    for(int d = 0; d < nextLayer.nodeSize; d++){
                        tempSum += nextLayer.partialDerivative[d][c] * nextLayer.w[d][c];
                    }
                    for(int d = 0; d < curLayer.partialDerivative[c].length; d++){
                        curLayer.partialDerivative[c][d] = tempSum * calcActivationDerivative(activationType, curLayer.h[c]);
                    }
                }
            }
        }
    }

    public static void updateWeight(List<Layer> layerList, double learnRate, int miniBatch) {
        for(int a = 1; a < layerList.size(); a++){
            Layer preLayer = layerList.get(a - 1);
            Layer curLayer = layerList.get(a);

            for(int c = 0; c < curLayer.nodeSize; c++){
                for(int d = 0; d < preLayer.nodeSize; d++){
                    double fullDerivative = curLayer.partialDerivative[c][d] * preLayer.h[d];
                    curLayer.w[c][d] -= learnRate * fullDerivative / miniBatch;
                }
            }
        }
    }

    public static void train(List<Layer> layerList, List<List<Double>> trainDataList, int epoch, int miniBatch, double learnRate, double lossMin, String activationType, String lossType) {
        int train = 0;
        double allLoss = Double.MAX_VALUE;
        while(train < epoch){
            Collections.shuffle(trainDataList);

            for(int a = 0; a < trainDataList.size(); a += miniBatch){
                double curLoss = 0.0;
                for(int c = a; c < a + miniBatch && c < trainDataList.size(); c++){
                    LoadInputToLayer(layerList, trainDataList, c);

                    forward(layerList, activationType);

                    curLoss += calcTotalLoss(layerList, getLabelList(trainDataList, c, layerList.get(0).nodeSize), lossType);
                    
                    backword(layerList, activationType, lossType, getLabelList(trainDataList, c, layerList.get(0).nodeSize));
                    
                    updateWeight(layerList, learnRate, miniBatch);
                }
                allLoss = curLoss / miniBatch;
            }

            train++;
            System.out.println("train: " + train + ", loss: " + allLoss);
            // printLayerList(layerList, train, allLoss);
        }
    }

    public static List<Double> getOutputList(List<Layer> layerList) {
        List<Double> res = new ArrayList<>();
        Layer outputLayer = layerList.get(layerList.size() - 1);
        for(int a = 0; a < outputLayer.h.length; a++){
            res.add(outputLayer.h[a]);
        }
        return res;
    }

    public static double calcPredict(List<Double> outputList) {
        double y_hat = outputList.get(0);
        return Math.abs(1 - y_hat) < Math.abs(y_hat - 0) ? 1.0 : 0.0;
    }

    public static List<Double> predict(List<Layer> layerList, List<List<Double>> testDataList, String activationType, int inputNodeSize) {
        double correct = 0.0;
        double wrong = 0.0;
        double total = 0.0;
        List<Double> predictList = new ArrayList<>();
        for(int a = 0; a < testDataList.size(); a++){
            LoadInputToLayer(layerList, testDataList, a);
            
            forward(layerList, activationType);
            
            List<Double> outputList = getOutputList(layerList);
            double output = outputList.get(0);

            double predict = calcPredict(outputList);
            predictList.add(predict);

            double label = -1.0;
            if(testDataList.get(a).size() > inputNodeSize){
                List<Double> labelList = getLabelList(testDataList, a, inputNodeSize);
                label = labelList.get(0);

                total++;
                if(label == predict){
                    correct++;
                }
                else{
                    wrong++;
                }
            }
            System.out.println("label: " + label + ", output: " + output + ", predict: " + predict);
        }

        System.out.println("total: " + total + ", correct: " + correct + ", wrong: " + wrong + ", accurate: " + (total == 0.0 ? total : correct / total));
        
        return predictList;
    }

    public static double L1(double y, double y_hat) {
        return Math.abs(y - y_hat);
    }

    public static double L2(double y, double y_hat) {
        return Math.pow(y - y_hat, 2.0) / 2.0;
    }

    public static double HuberRobust(double y, double y_hat) {
        if(Math.abs(y - y_hat) > 1){
            return Math.abs(y - y_hat) - 0.5;
        }
        else{
            return Math.pow(y - y_hat, 2.0) / 2.0;
        }
    }

    public static double Sigmoid(double y_hat) {
        return 1.0 / (1.0 + Math.exp(-y_hat));
    }

    public static double SigmoidDerivative(double y_hat) {
        return y_hat * (1.0 - y_hat);
    }

    public static double Tanh(double y_hat) {
        return Math.tanh(y_hat);
    }

    public static double TanhDerivative(double y_hat) {
        return 1.0 - Math.tanh(y_hat) * Math.tanh(y_hat);
    }

    public static double ReLU(double y_hat) {
        return Math.max(0, y_hat);
    }

    public static double ReLUDerivative(double y_hat) {
        return y_hat < 0 ? 0 : 1;
    }

    public static double CrossEntropy(double y, double y_hat) {
        return -y * Math.log(y_hat);
    }

    public static double CrossEntropyDerivative(double y, double y_hat) {
        return - y / y_hat;
    }

    public static double mse(double y, double y_hat) {
        return Math.pow(y - y_hat, 2.0) / 2.0;
    }

    public static double mseDerivative(double y, double y_hat) {
        return - (y - y_hat);
    }

    public static double calcTotalLoss(List<Layer> layerList, List<Double> labelList, String lossType) {
        Layer outputLayer = layerList.get(layerList.size() - 1);
        double totalLoss = 0.0;
        for(int a = 0; a < outputLayer.h.length; a++){
            totalLoss += calcLoss(lossType, labelList.get(a), outputLayer.h[a]);
        }
        return totalLoss;
    }

    public static double calcLoss(String lossType, double y, double y_hat) {
        switch (lossType) {
            case "mse":
                return mse(y, y_hat);

            case "cross":
                return CrossEntropy(y, y_hat);

            default:
                return mse(y, y_hat);
        }
    }

    public static double calcLossDerivative(String lossType, double y, double y_hat) {
        switch (lossType) {
            case "mse":
                return mseDerivative(y, y_hat);

            case "cross":
                return CrossEntropyDerivative(y, y_hat);

            default:
                return mseDerivative(y, y_hat);
        }
    }

    public static double calcActivation(String activationType, double y_hat) {
        switch (activationType) {
            case "sigmoid":
                return Sigmoid(y_hat);

            case "tanh":
                return Tanh(y_hat);

            case "relu":
                return ReLU(y_hat);

            default:
                return Tanh(y_hat);
        }
    }

    public static double calcActivationDerivative(String activationType, double y_hat) {
        switch (activationType) {
            case "sigmoid":
                return SigmoidDerivative(y_hat);

            case "tanh":
                return TanhDerivative(y_hat);

            case "relu":
                return ReLUDerivative(y_hat);

            default:
                return TanhDerivative(y_hat);
        }
    }

    public static double nextDoubleWithinRange(Random random, double randomMin, double randomMax) {
        return randomMin + (randomMax - randomMin) * random.nextDouble();
    }

    public static double[] initialize1DArray(int nodeSize, Random random, double randomMin, double randomMax) {
        double[] res = new double[nodeSize];
        for(int a = 0; a < res.length; a++){
            res[a] = nextDoubleWithinRange(random, randomMin, randomMax);
        }
        return res;
    }

    public static double[][] initialize2DArray(int preNodeSize, int curNodeSize, Random random, double randomMin, double randomMax) {
        double[][] res = new double[curNodeSize][preNodeSize];
        for(int a = 0; a < res.length; a++){
            res[a] = initialize1DArray(preNodeSize, random, randomMin, randomMax);
        }
        return res;
    }

    public static List<Double> getLabelList(List<List<Double>> dataList, int dataIndex, int inputNodeSize) {
        List<Double> temp = dataList.get(dataIndex);
        List<Double> res = new ArrayList<>();
        for(int a = inputNodeSize; a < temp.size(); a++){
            res.add(temp.get(a));
        }
        return res;
    }

    public static void LoadInputToLayer(List<Layer> layerList, List<List<Double>> trainDataList, int trainDataIndex) {
        Layer inputLayer = layerList.get(0);
        List<Double> dataList = trainDataList.get(trainDataIndex);
        for(int a = 0; a < inputLayer.h.length; a++){
            inputLayer.h[a] = dataList.get(a);
        }
    }

    public static void readInput(String args[], List<List<Double>> trainDataList, List<List<Double>> testDataList) {
        try (
            Scanner trainDataScanner = new Scanner(new File(args[0]));
            Scanner trainLabelScanner = new Scanner(new File(args[1]));
            Scanner testDataScanner = new Scanner(new File(args[2]));
            Scanner testLabelScanner = args.length >= 4 ? new Scanner(new File(args[3])) : null;
        ) 
        {
            while(trainDataScanner.hasNext() && trainLabelScanner.hasNext()){
                String dataLine = trainDataScanner.nextLine();
                String[] dataArray = dataLine.split(",");
                
                List<Double> dataList = new ArrayList<>();
                for(int a = 0; a < dataArray.length; a++){
                    dataList.add(Double.valueOf(dataArray[a]));
                }

                double x = Double.valueOf(dataArray[0]);
                double y = Double.valueOf(dataArray[1]);

                dataList.add(x * y);
                dataList.add(Math.sin(x));
                dataList.add(Math.sin(y));

                String labelLine = trainLabelScanner.nextLine();
                String[] labelArray = labelLine.split(",");

                for(int a = 0; a < labelArray.length; a++){
                    dataList.add(Double.valueOf(labelArray[a]));
                }

                trainDataList.add(dataList);
            }
            
            while(testDataScanner.hasNext()){
                String dataLine = testDataScanner.nextLine();
                String[] dataArray = dataLine.split(",");

                List<Double> dataList = new ArrayList<>();
                for(int a = 0; a < dataArray.length; a++){
                    dataList.add(Double.valueOf(dataArray[a]));
                }

                double x = Double.valueOf(dataArray[0]);
                double y = Double.valueOf(dataArray[1]);

                dataList.add(x * y);
                dataList.add(Math.sin(x));
                dataList.add(Math.sin(y));

                if(testLabelScanner != null && testLabelScanner.hasNext()){
                    String labelLine = testLabelScanner.nextLine();
                    String[] labelArray = labelLine.split(",");

                    for(int a = 0; a < labelArray.length; a++){
                        dataList.add(Double.valueOf(labelArray[a]));
                    }
                }

                testDataList.add(dataList);
            }
        } 
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void writePredict(List<Double> predicList, String predictFileName) {
        try (
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(predictFileName));
        ) 
        {
            for(int a = 0; a < predicList.size(); a++){
                bufferedWriter.write(predicList.get(a).intValue() + "");
                bufferedWriter.newLine();
            }
        } 
        catch (Exception e) {

        }
    }

    public static void printLayerList(List<Layer> layerList, int train, double allLoss) {
        System.out.println("==========train " + train + "==========");
        System.out.println("train: " + train + ", loss: " + allLoss);
        for(int a = 0; a < layerList.size(); a++){
            Layer curLayer = layerList.get(a);
            
            System.out.println("layerType: " + curLayer.layerType + ", nodeSize: " + curLayer.nodeSize);
            
            System.out.println("double[][] w: " + (curLayer.w == null ? "null" : ""));
            print2DArray(curLayer.w);
            
            System.out.println("double[] b: " + (curLayer.b == null ? "null" : ""));
            print1DArray(curLayer.b);
            
            System.out.println("double[][] partialDerivative: " + (curLayer.partialDerivative == null ? "null" : ""));
            print2DArray(curLayer.partialDerivative);
            
            System.out.println("double[] z: " + (curLayer.z == null ? "null" : ""));
            print1DArray(curLayer.z);
            
            System.out.println("double[] h: " + (curLayer.h == null ? "null" : ""));
            print1DArray(curLayer.h);
        }
        System.out.println("==============================");
    }

    public static void print1DArray(double[] temp) {
        if(temp != null){
            System.out.println(Arrays.toString(temp));
        }
    }

    public static void print2DArray(double[][] temp) {
        if(temp != null){
            for(int a = 0; a < temp.length; a++){
                print1DArray(temp[a]);
            }
        }
    }

    public static void printInput(List<List<Double>> dataList, String dataSetType) {
        System.out.println("==========" + dataSetType + "==========");
        for(int a = 0; a < dataList.size(); a++){
            System.out.println(dataList.get(a));
        }
        System.out.println("==============================");
    }
}

class Layer{
    
    String layerType;
    int nodeSize;
    double[][] w;
    double[] b;
    double[][] partialDerivative;
    double[] z;
    double[] h;

    public Layer(String layerType, int nodeSize, double[][] w, double[] b, double[][] partialDerivative, double[] z, double[] h){
        this.layerType = layerType;
        this.nodeSize = nodeSize;
        this.w = w;
        this.b = b;
        this.partialDerivative = partialDerivative;
        this.z = z;
        this.h = h;
    }

}