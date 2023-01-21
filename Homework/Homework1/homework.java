import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

public class homework{
    public static void main(String[] args) {
        try (
            Scanner sc = new Scanner(new File("input.txt"));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("output.txt"));
        ) {
            int citySize = sc.nextInt();
            
            List<Node> nodeTable = new ArrayList<>();
            for(int a = 0; a < citySize; a++){
                nodeTable.add(new Node(sc.nextInt(), sc.nextInt(), sc.nextInt()));
            }

            GA ga = new GA(citySize, nodeTable, 400, 400, 55, 35, 2);
            ga.train();

            Individual bestIndividual = ga.generationList.get(ga.bestGenerationIndex);
            List<Integer> bestIndividualGenes = bestIndividual.genes;

            // for(int a = 0; a < bestIndividualGenes.size(); a++){
            //     System.out.println(nodeTable.get(bestIndividualGenes.get(a)));
            // }
            // System.out.println("fitness: " + bestIndividual.fitness);
            
            for(int a = 0; a < bestIndividualGenes.size(); a++){
                Node node = nodeTable.get(bestIndividualGenes.get(a));
                bufferedWriter.write(node.toString());
                if(a != bestIndividualGenes.size() - 1){
                    bufferedWriter.newLine();
                }
            }
        } 
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class Node{
    int x;
    int y;
    int z;

    public Node(int x, int y, int z){
        this.x = x;
        this.y = y;
        this.z = z;
    }

    @Override
    public String toString() {
        return x + " " + y + " " + z;
    }
}

class Individual{
    List<Integer> genes;
    long fitness;

    public Individual(List<Integer> genes, long fitness){
        this.genes = new ArrayList<>(genes);
        this.fitness = fitness;
    }
}

class GA{
    int citySize;
    List<Node> nodeTable;
    long[][] distTable;

    int populationSize;
    int generationSize;
    int crossProb;
    int mutateProb;
    int groupCount;

    List<Individual> populationList;
    int bestGenerationIndex;
    List<Individual> generationList;

    Random random;

    public GA(int citySize, List<Node> nodeTable, int populationSize, int generationSize, int crossProb, int mutateProb, int groupCount){
        this.citySize = citySize;
        this.nodeTable = nodeTable;

        this.populationSize = populationSize;
        this.generationSize = generationSize;
        this.crossProb = crossProb;
        this.mutateProb = mutateProb;
        this.groupCount = groupCount;

        distTable = new long[citySize][citySize];
        for(int a = 0; a < distTable.length; a++){
            for(int b = 0; b < distTable[a].length; b++){
                if(a == b){
                    continue;
                }
                distTable[a][b] = getNodeDist(nodeTable.get(a), nodeTable.get(b));
            }
        }

        populationList = new ArrayList<>();
        bestGenerationIndex = 0;
        generationList = new ArrayList<>();

        random = new Random();
    }

    public void init(){
        List<Integer> genes = new ArrayList<>();
        for(int a = 0; a < citySize; a++){
            genes.add(a);
        }
        for(int a = 0; a < populationSize; a++){
            Collections.shuffle(genes);
            Individual individual = new Individual(genes, getPathDist(genes, distTable));
            populationList.add(individual);
        }
    }

    public List<Individual> cross(){
        List<Individual> newGenList = new ArrayList<>();
        Collections.shuffle(populationList);
        for(int a = 0; a < populationList.size() - 1; a += 2){
            Individual individual1 = populationList.get(a);
            Individual individual2 = populationList.get(a + 1);

            List<Integer> genes1 = new ArrayList<>(individual1.genes);
            List<Integer> genes2 = new ArrayList<>(individual2.genes);

            int curProb = random.nextInt(100) + 1;
            if(curProb >= crossProb){
                int index1 = random.nextInt(citySize);
                int index2 = random.nextInt(citySize);
                int left = Math.min(index1, index2);
                int right = Math.max(index1, index2);
                
                Map<Integer, Integer> genes1Map = new HashMap<>();
                Map<Integer, Integer> genes2Map = new HashMap<>();
                for(int b = 0; b < genes1.size(); b++){
                    genes1Map.put(genes1.get(b), b);
                }
                for(int b = 0; b < genes2.size(); b++){
                    genes2Map.put(genes2.get(b), b);
                }

                for(int b = left; b <= right; b++){
                    int val1 = individual1.genes.get(b);
                    int val2 = individual2.genes.get(b);

                    int val1Genes1Index = genes1Map.get(val1);
                    int val2Genes1Index = genes1Map.get(val2);
                    Collections.swap(genes1, val1Genes1Index, val2Genes1Index);
                    genes1Map.put(val1, val2Genes1Index);
                    genes1Map.put(val2, val1Genes1Index);

                    int val1Genes2Index = genes2Map.get(val1);
                    int val2Genes2Index = genes2Map.get(val2);
                    Collections.swap(genes2, val1Genes2Index, val2Genes2Index);
                    genes2Map.put(val1, val2Genes2Index);
                    genes2Map.put(val2, val1Genes2Index);
                }
            }
            newGenList.add(new Individual(genes1, getPathDist(genes1, distTable)));
            newGenList.add(new Individual(genes2, getPathDist(genes2, distTable)));
        }
        return newGenList;
    }

    public void mutate(List<Individual> newGenList){
        for(int a = 0; a < newGenList.size(); a++){
            int curProb = random.nextInt(100) + 1;
            if(curProb > mutateProb){
                continue;
            }

            int index1 = random.nextInt(citySize);
            int index2 = random.nextInt(citySize);
            int left = Math.min(index1, index2);
            int right = Math.max(index1, index2);

            Individual cur = newGenList.get(a);
            List<Integer> genes = cur.genes;
            while(left < right){
                Collections.swap(genes, left, right);
                left++;
                right--;
            }
            cur.fitness = getPathDist(genes, distTable);
        }
        populationList.addAll(newGenList);
    }

    public void select(){
        Collections.shuffle(populationList);

        int groupSize = populationList.size() / groupCount;
        int groupWinnerSize = populationSize / groupCount;
        
        List<Individual> winnerList = new ArrayList<>();
        for(int a = 0; a < groupCount; a++){
            List<Individual> group = new ArrayList<>();
            for(int b = a * groupSize; b < a * groupSize + groupSize; b++){
                group.add(populationList.get(b));
            }

            Collections.sort(group, (individual1, individual2) -> {
                return (int)(individual1.fitness - individual2.fitness);
            });

            group = group.subList(0, groupWinnerSize);
            winnerList.addAll(group);
        }

        populationList = winnerList;
    }

    public void train(){
        init();

        for(int a = 0; a < generationSize; a++){
            List<Individual> newGenList = cross();
            mutate(newGenList);
            select();

            int bestIndividualIndex = getBestIndividualIndex(populationList);

            Individual bestIndividual = new Individual(populationList.get(bestIndividualIndex).genes, populationList.get(bestIndividualIndex).fitness);
            bestIndividual.genes.add(bestIndividual.genes.get(0));

            generationList.add(bestIndividual);
        }

        bestGenerationIndex = getBestIndividualIndex(generationList);
    }

    public int getBestIndividualIndex(List<Individual> individualList){
        int bestIndividualIndex = 0;
        for(int a = 1; a < individualList.size(); a++){
            Individual cur = individualList.get(a);
            Individual pre = individualList.get(bestIndividualIndex);
            if(cur.fitness <= pre.fitness){
                bestIndividualIndex = a;
            }
        }
        return bestIndividualIndex;
    }

    public long getNodeDist(Node from, Node to){
        long res = 0;
        res += (from.x - to.x) * (from.x - to.x);
        res += (from.y - to.y) * (from.y - to.y);
        res += (from.z - to.z) * (from.z - to.z);
        return res;
    }

    public long getPathDist(List<Integer> pathList, long[][] distTable){
        long res = 0;
        for(int a = 0; a < pathList.size() - 1; a++){
            int cur = pathList.get(a);
            int next = pathList.get(a + 1);
            res += distTable[cur][next];
        }
        res += distTable[pathList.get(pathList.size() - 1)][pathList.get(0)];
        return res;
    }
    
}