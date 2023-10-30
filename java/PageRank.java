
//Part 1 Code
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class PageRank {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: WikipediaPageRank <path_to_links_file> <path_to_titles_file> <output_directory>");
            System.exit(1);
        }

        String linksFilePath = args[0];
        String titlesFilePath = args[1];
        String outputPath = args[2];

        SparkSession spark = SparkSession
                .builder()
                .appName("WikipediaPageRank")
                .getOrCreate();

        // Obtain JavaSparkContext from SparkSession's SparkContext
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(spark.sparkContext());

        // Load titles from titles-sorted.txt
        JavaRDD<String> titles = spark.read().textFile(titlesFilePath).javaRDD();
        JavaPairRDD<String, String> titleIndex = titles.zipWithIndex().mapToPair(ti -> new Tuple2<>(String.valueOf(ti._2() + 1), ti._1())); // Adjusting for 1-based index

        // Loads in links file
        JavaRDD<String> lines = spark.read().textFile(linksFilePath).javaRDD();

        // Parsing the data to match the format
        JavaPairRDD<String, Iterable<String>> links = lines.mapToPair(s -> {
            String[] parts = s.split(":");
            String[] destinations = parts[1].trim().split("\\s+");
            List<String> destList = new ArrayList<>();
            for (String dest : destinations) {
                destList.add(dest.trim());
            }
            return new Tuple2<>(parts[0].trim(), destList);
        });

        long totalPages = links.count();
        JavaPairRDD<String, Double> ranks = links.mapValues(v -> 1.0 / totalPages);

        for (int i = 0; i < 25; i++) {
            JavaPairRDD<String, Double> tempRank = links.join(ranks).values().flatMapToPair(t -> {
                List<Tuple2<String, Double>> results = new ArrayList<>();
                List<String> urls = new ArrayList<>();
                t._1.forEach(urls::add);
                double rankPerURL = t._2 / urls.size();
                for (String url : urls) {
                    results.add(new Tuple2<>(url, rankPerURL));
                }
                return results.iterator();
            });

            ranks = tempRank.reduceByKey((a, b) -> a + b);
        }


        // Joining the ranks with the titles
        JavaPairRDD<Double, String> swappedRanks = titleIndex.join(ranks).mapToPair(item -> new Tuple2<>(item._2._2, item._2._1));

        // Sort and save the results to a file
        JavaPairRDD<Double, String> sortedRanks = swappedRanks.sortByKey(false);
        JavaPairRDD<String, Double> finalOutput = sortedRanks.mapToPair(item -> new Tuple2<>(item._2, item._1));
        JavaPairRDD<String, Double> top10RDD = jsc.parallelizePairs(finalOutput.take(10), 1);
        top10RDD.coalesce(1).saveAsTextFile(outputPath);

        spark.stop();
    }
}
