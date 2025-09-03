/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class PQTrainingWithSiftBenchmark {
    private static final Logger log = LoggerFactory.getLogger(PQTrainingWithSiftBenchmark.class);
    private RandomAccessVectorValues ravv;
    private List<VectorFloat<?>> baseVectors;
    private List<VectorFloat<?>> queryVectors;
    private List<List<Integer>> groundTruth;
    @Param({"16", "32", "64"})
    private int M; // Number of subspaces
    int originalDimension;

    @Setup
    public void setup() throws IOException {
        var siftPath = "siftsmall";
        baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        log.info("base vectors size: {}, query vectors size: {}, loaded, dimensions {}",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());
        originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
    }

    @Benchmark
    public void productQuantizationComputeBenchmark(Blackhole blackhole) throws IOException {
        // Compress the original vectors using PQ. this represents a compression ratio of 128 * 4 / 16 = 32x
        ProductQuantization pq = ProductQuantization.compute(ravv,
                M, // number of subspaces
                256, // number of centroids per subspace
                true); // center the dataset

        blackhole.consume(pq);
    }
}
