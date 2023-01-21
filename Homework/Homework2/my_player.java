import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class my_player{
    public static void main(String[] args) {
        try (
            Scanner sc = new Scanner(new File("input.txt"));
            BufferedWriter posBufferedWriter = new BufferedWriter(new FileWriter("output.txt"));
        ) 
        {
            long startTime = System.currentTimeMillis();

            int boardSize = 5;
            int maxStep = boardSize * boardSize;
            int maxDepth = 6;
            int[][] orthogonalDirs = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
            int[][] diagonalDirs = new int[][]{{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
            
            char myPlayer = sc.next().charAt(0);
            StringBuilder preBoard = new StringBuilder();
            StringBuilder curBoard = new StringBuilder();
            for(int a = 0; a < boardSize; a++){
                preBoard.append(sc.next());
            }
            for(int a = 0; a < boardSize; a++){
                curBoard.append(sc.next());
            }
            int curStep = readWriteStep(preBoard, myPlayer);
            int curDepth = 0;

            Double[] res = getNextMove(preBoard, curBoard, boardSize, maxStep, maxDepth, curStep, curDepth, myPlayer, myPlayer, orthogonalDirs, diagonalDirs);
            System.out.println("myOutput: " + Arrays.toString(res));

            if(res[0] < 0 || res[1] < 0){
                posBufferedWriter.write("PASS");
            }
            else{
                posBufferedWriter.write(res[0].intValue() + "," + res[1].intValue());
            }
            posBufferedWriter.newLine();

            System.out.println("time: " + (System.currentTimeMillis() - startTime));
        } 
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Double[] getNextMove(StringBuilder preBoard, StringBuilder curBoard, int boardSize, int maxStep, int maxDepth, int curStep, int curDepth, char myPlayer, char curPlayer, int[][] orthogonalDirs, int[][] diagonalDirs) {
        if(curStep == 1){
            return new Double[]{2.0, 2.0, 0.0};
        }
        if(curStep == 2){
            int boardIndex = getIndexByPosition(2, 2, boardSize);
            if(curBoard.charAt(boardIndex) == '0'){
                return new Double[]{2.0, 2.0, 0.0};
            }
            return new Double[]{2.0, 3.0, 0.0};
        }

        Double[] resPos = new Double[]{-1.0, -1.0, 0.0};

        List<MyState> myStateList = getPossibleState(preBoard, curBoard, boardSize, myPlayer, orthogonalDirs, diagonalDirs);
        if(myStateList.isEmpty()){
            return resPos;
        }
        double[][] rewardTable = getRewardTable();
        
        double resMax = -Double.MAX_VALUE;
        for(MyState myState : myStateList){
            int libertyNear = countLibertyNear(myState.newBoard, boardSize, myState.position[0], myState.position[1], myState.curChessType, orthogonalDirs);
            int diagonalChessSize = countDiagnalChess(myState.newBoard, boardSize, myState.position[0], myState.position[1], myState.curChessType, diagonalDirs);
            double nextVal = getNextMoveValue(myState.curBoard, myState.newBoard, boardSize, maxStep, maxDepth, curStep + 1, curDepth + 1, myPlayer, getRivalType(curPlayer), -Double.MAX_VALUE, Double.MAX_VALUE, orthogonalDirs, diagonalDirs);
            double curVal = nextVal + libertyNear + diagonalChessSize + getReward(rewardTable, myState.position[0], myState.position[1]);

            if(curVal > resMax){
                resPos[0] = myState.position[0] * 1.0;
                resPos[1] = myState.position[1] * 1.0;
                resPos[2] = curVal;
                resMax = curVal;
            }
        }
        return resPos;
    }

    public static Double getNextMoveValue(StringBuilder preBoard, StringBuilder curBoard, int boardSize, int maxStep, int maxDepth, int curStep, int curDepth, char myPlayer, char curPlayer, double alpha, double beta, int[][] orthogonalDirs, int[][] diagonalDirs){
        if(curStep >= maxStep || curDepth >= maxDepth){
            return evalScore(curBoard, boardSize, myPlayer);
        }
        
        List<MyState> myStateList = getPossibleState(preBoard, curBoard, boardSize, curPlayer, orthogonalDirs, diagonalDirs);
        if(myStateList.isEmpty()){
            return evalScore(curBoard, boardSize, curPlayer);
        }

        double curMax = -Double.MAX_VALUE;
        double curMin = Double.MAX_VALUE;
        for(MyState myState : myStateList){
            double nextVal = getNextMoveValue(myState.curBoard, myState.newBoard, boardSize, maxStep, maxDepth, curStep + 1, curDepth + 1, myPlayer, getRivalType(curPlayer), alpha, beta, orthogonalDirs, diagonalDirs);
            
            if(myPlayer == curPlayer){
                if(curMax < nextVal){
                    curMax = nextVal;
                }
                if(beta <= curMax){
                    return curMax;
                }
                if(alpha < curMax){
                    alpha = curMax;
                }
            }
            else{
                if(curMin > nextVal){
                    curMin = nextVal;
                }
                if(alpha >= curMin){
                    return curMin;
                }
                if(beta > curMin){
                    beta = curMin;
                }
            }
        }
        return myPlayer == curPlayer ? curMax : curMin;
    }

    public static double evalScore(StringBuilder curBoard, int boardSize, char myPlayer) {
        return 10.0 * (countChess(curBoard, boardSize, myPlayer) - countChess(curBoard, boardSize, getRivalType(myPlayer)));
    }

    public static double countChess(StringBuilder curBoard, int boardSize, char curChessType) {
        double count = 0.0;
        for(int a = 0; a < curBoard.length(); a++){
            char c = curBoard.charAt(a);
            if(c == curChessType){
                count++;
            }
        }
        if(curChessType == '2'){
            count += boardSize / 2.0;
        }
        return count;
    }

    public static int countLibertyNear(StringBuilder curBoard, int boardSize, int row, int col, char curChessType, int[][] orthogonalDirs) {
        int count = 0;
        for(int[] dir : orthogonalDirs){
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if(newRow < 0 || newCol < 0 || newRow >= boardSize || newCol >= boardSize){
                continue;
            }
            int newIndex = getIndexByPosition(newRow, newCol, boardSize);
            if(curBoard.charAt(newIndex) == '0'){
                count++;
            }
        }
        return count;
    }

    public static int countLiberty(StringBuilder curBoard, int boardSize, int row, int col, char curChessType, Set<Integer> visited, int[][] orthogonalDirs){
        if(row < 0 || col < 0 || row >= boardSize || col >= boardSize 
        || getRivalType(curChessType) == curBoard.charAt(getIndexByPosition(row, col, boardSize)) 
        || visited.contains(getIndexByPosition(row, col, boardSize)))
        {
            return 0;
        }
        if(curBoard.charAt(getIndexByPosition(row, col, boardSize)) == '0'){
            return 1;
        }
        int res = 0;
        visited.add(getIndexByPosition(row, col, boardSize));
        for(int[] dir : orthogonalDirs){
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            res += countLiberty(curBoard, boardSize, newRow, newCol, curChessType, visited, orthogonalDirs);
        }
        return res;
    }

    public static Map<Integer, Integer> countChessAllLiberty(StringBuilder curBoard, int boardSize, char curChessType, int[][] orthogonalDirs){
        Map<Integer, Integer> map = new HashMap<>();
        Set<Integer> visited = new HashSet<>();
        for(int a = 0; a < curBoard.length(); a++){
            char c = curBoard.charAt(a);
            if(c == curChessType && !visited.contains(a)){
                Set<Integer> tempVisited = new HashSet<>();
                int[] position = getPositionByIndex(a, boardSize);
                int liberty = countLiberty(curBoard, boardSize, position[0], position[1], curChessType, tempVisited, orthogonalDirs);
                map.put(liberty, map.getOrDefault(liberty, 0) + tempVisited.size());
                visited.addAll(tempVisited);
            }
        }
        return map;
    }

    public static Set<Integer> getCaptureChess(StringBuilder curBoard, int boardSize, int row, int col, char curChessType, int[][] orthogonalDirs){
        Set<Integer> captureChess = new HashSet<>();
        for(int[] dir : orthogonalDirs){
            Set<Integer> tempVisited = new HashSet<>();
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if(0 == countLiberty(curBoard, boardSize, newRow, newCol, getRivalType(curChessType), tempVisited, orthogonalDirs)){
                captureChess.addAll(tempVisited);
            }
        }
        return captureChess;
    }

    public static void removeCaptureChess(StringBuilder curBoard, Set<Integer> captureChess){
        for(int index : captureChess){
            curBoard.setCharAt(index, '0');
        }
    }

    public static boolean isSameBoard(StringBuilder preBoard, StringBuilder curBoard){
        return preBoard.toString().equals(curBoard.toString());
    }

    public static int getIndexByPosition(int row, int col, int boardSize){
        return row * boardSize + col;
    }

    public static int[] getPositionByIndex(int index, int boardSize){
        return new int[]{index / boardSize, index % boardSize};
    }

    public static char getRivalType(char curType){
        return '1' == curType ? '2' : '1';
    }

    public static StringBuilder getCopyBoard(StringBuilder curBoard){
        return new StringBuilder(curBoard.toString());
    }

    public static double[][] getRewardTable() {
        return new double[][] {
            {-2.0, 0.0, 0.5, 0.0, -2.0},
            {0.0, 0.5, 1.0, 0.5, 0.0},
            {0.5, 1.0, 2.0, 1.0, 0.5},
            {0.0, 0.5, 1.0, 0.5, 0.0},
            {-2.0, 0.0, 0.5, 0.0, -2.0}
        };
    }

    public static double getReward(double[][] rewardTable, int row, int col) {
        return rewardTable[row][col];
    }

    public static int countDiagnalChess(StringBuilder curBoard, int boardSize, int row, int col, char curChessType, int[][] diagonalDirs) {
        int count = 0;
        for(int[] dir : diagonalDirs){
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if(newRow < 0 || newCol < 0 || newRow >= boardSize || newCol >= boardSize){
                continue;
            }
            int newIndex = getIndexByPosition(newRow, newCol, boardSize);
            if(curBoard.charAt(newIndex) == curChessType){
                count++;
            }
        }
        return count;
    }

    public static MyState getValidState(StringBuilder preBoard, StringBuilder curBoard, int boardSize, int boardIndex, char curChessType, int[][] orthogonalDirs) {
        int[] position = getPositionByIndex(boardIndex, boardSize);
        int row = position[0];
        int col = position[1];
        if(row < 0 || col < 0 || row >= boardSize || col >= boardSize || curBoard.charAt(boardIndex) != '0'){
            return null;
        }
        StringBuilder newBoard = getCopyBoard(curBoard);
        newBoard.setCharAt(boardIndex, curChessType);
        int liberty = countLiberty(newBoard, boardSize, row, col, curChessType, new HashSet<>(), orthogonalDirs);
        Set<Integer> captureChess = getCaptureChess(newBoard, boardSize, row, col, curChessType, orthogonalDirs);
        removeCaptureChess(newBoard, captureChess);
        if(liberty == 0 && (captureChess.isEmpty() || isSameBoard(preBoard, newBoard))){
            return null;
        }
        return new MyState(getCopyBoard(curBoard), getCopyBoard(newBoard), boardIndex, Arrays.copyOf(position, 2), curChessType, liberty, new HashSet<>(captureChess));
    }

    public static boolean hasNearChess(StringBuilder curBoard, int boardSize, int boardIndex, int[][] orthogonalDirs, int[][] diagonalDirs) {
        int[] position = getPositionByIndex(boardIndex, boardSize);
        int row = position[0];
        int col = position[1];
        for(int[] dir : orthogonalDirs){
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if(newRow < 0 || newCol < 0 || newRow >= boardSize || newCol >= boardSize){
                continue;
            }
            int newIndex = getIndexByPosition(newRow, newCol, boardSize);
            if(curBoard.charAt(newIndex) != '0'){
                return true;
            }
        }
        for(int[] dir : diagonalDirs){
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            if(newRow < 0 || newCol < 0 || newRow >= boardSize || newCol >= boardSize){
                continue;
            }
            int newIndex = getIndexByPosition(newRow, newCol, boardSize);
            if(curBoard.charAt(newIndex) != '0'){
                return true;
            }
        }
        return false;
    }

    public static List<MyState> getPossibleState(StringBuilder preBoard, StringBuilder curBoard, int boardSize, char curChessType, int[][] orthogonalDirs, int[][] diagonalDirs) {
        List<MyState> myStateList = new ArrayList<>();
        for(int a = 0; a < curBoard.length(); a++){
            MyState myState = null;
            if(hasNearChess(curBoard, boardSize, a, orthogonalDirs, diagonalDirs) && (myState = getValidState(preBoard, curBoard, boardSize, a, curChessType, orthogonalDirs)) != null){
                myStateList.add(myState);
            }
        }
        return myStateList;
    }

    public static int readWriteStep(StringBuilder preBoard, char myPlayer) {
        StringBuilder initialBoard = new StringBuilder();
        for(int a = 0; a < preBoard.length(); a++){
            initialBoard.append('0');
        }
        int curStep = 0;
        if(isSameBoard(initialBoard, preBoard)){
            curStep = myPlayer - '0';
        }
        else{
            try (
                Scanner sc = new Scanner(new File("mystep.txt"));
            ) 
            {
                curStep = sc.nextInt();
            } 
            catch (Exception e) {
                e.printStackTrace();
            }
        }
        try (
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("mystep.txt"));
        ) 
        {
            bufferedWriter.write((curStep + 2) + "");
            bufferedWriter.newLine();
        } 
        catch (Exception e) {
            e.printStackTrace();
        }
        return curStep;
    }
}

class MyState{
    StringBuilder curBoard;
    StringBuilder newBoard;
    int boardIndex;
    int[] position;
    char curChessType;
    int liberty;
    Set<Integer> captureChess;

    public MyState(StringBuilder curBoard, StringBuilder newBoard, int boardIndex, int[] position, char curChessType, int liberty, Set<Integer> captureChess){
        this.curBoard = curBoard;
        this.newBoard = newBoard;
        this.boardIndex = boardIndex;
        this.position = position;
        this.curChessType = curChessType;
        this.liberty = liberty;
        this.captureChess = captureChess;
    }
}