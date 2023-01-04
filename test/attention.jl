@testset "different batchsizes" begin
    n = 15
    lenq = 3
    lenkv = 4
    for batch_size in [(), 1, 2, (2,1,3)], num_heads in [1, 3, 5]
        q = rand(Float32, n, lenq, batch_size...)
        k = rand(Float32, n, lenkv, batch_size...)
        v = rand(Float32, n, lenkv, batch_size...)
        y, α = dot_product_attention(q, k, v; num_heads)
        @test y isa Array{Float32}
        @test size(y) == (n, lenq, batch_size...)
        @test size(α) == (lenkv, lenq, num_heads, batch_size...)
        @test sum(α, dims=1) ≈ ones(1, lenq, num_heads, batch_size...)
    end
end

@testset "dot_product_attention_scores" begin
    q = k = reshape([1:24;], 4, 2, 3, 1) ./ 24
    α = dot_product_attention_scores(q, k)
    q2, k2 = reshape.((q, k), 8, 3, 1)
    y, α2 = dot_product_attention(q2, k2, k2; num_heads=2)
    @test α ≈ α2
end

@testset "specific results" begin
    q = k = v = reshape([1:12;], 4, 3, 1) ./ 12
    y, α = dot_product_attention(q, k, v; num_heads=2)
    @test y ≈ [0.4297536645089624 0.46431026790247376 0.49773020657887745; 0.5130869978422957 0.5476436012358071 0.5810635399122107; 0.6137914555895531 0.6478764227436047 0.6804545876711346; 0.6971247889228864 0.731209756076938 0.763787921004468;;;]
    @test α ≈ [0.3138955704910261 0.264431440679808 0.21921458153690657; 0.3329478654910607 0.32820631493296265 0.31838021718955445; 0.35315656401791323 0.4073622443872293 0.4624052012735389;;; 0.2886914482847165 0.24123865285082136 0.19843756756539277; 0.33124273666190807 0.3238934260675431 0.31176110185581074; 0.3800658150533755 0.43486792108163547 0.4898013305787966;;;;]
end

@testset "mask" begin
    q = rand(4, 2, 3, 1)
    k = rand(4, 2, 5, 1)
    mask = rand(Bool, (5, 3))
    α = dot_product_attention_scores(q, k; mask)
    @test all((α[:,:,1,1].> 0) .== mask)
    @test all((α[:,:,2,1].> 0) .== mask)
end