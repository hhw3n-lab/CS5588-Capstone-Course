// ==============================================================================
// Retail360: GraphRAG for Evidence-Grounded 60-Day Repurchase Prediction
// Author: Harris Hamid | CS 5588 Data Science Capstone | Hack-A-Roo 2026
// Target: Neo4j Aura Cloud (`fb2e45c1.databases.neo4j.io`)
// ==============================================================================

// ------------------------------------------------------------------------------
// 1. Constraints & Indexes
// ------------------------------------------------------------------------------
CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT invoice_id_unique IF NOT EXISTS FOR (i:Invoice) REQUIRE i.id IS UNIQUE;
CREATE CONSTRAINT product_code_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.stockCode IS UNIQUE;
CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (comm:Community) REQUIRE comm.id IS UNIQUE;
CREATE CONSTRAINT experiment_name_unique IF NOT EXISTS FOR (e:Experiment) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT tool_name_unique IF NOT EXISTS FOR (t:Tool) REQUIRE t.name IS UNIQUE;

// ------------------------------------------------------------------------------
// 2. Communities (Modularity-based Product Clusters)
// ------------------------------------------------------------------------------
MERGE (c17:Community {id: "C17"})
ON CREATE SET c17.theme = "Home Decor & Ambient Lighting",
              c17.detectionAlgorithm = "Leiden Modularity Clustering",
              c17.description = "Groups products that frequently co-occur on the same invoices (e.g., C17 = Home Decor). Captures latent purchase affinity beyond explicit categories.";

MERGE (c08:Community {id: "C08"})
ON CREATE SET c08.theme = "Bags & Storage Solutions",
              c08.detectionAlgorithm = "Leiden Modularity Clustering",
              c08.description = "Bags and storage co-purchase cluster (e.g., C08 = Bags).";

// ------------------------------------------------------------------------------
// 3. Products (StockCodes & Poster Aliases C17-P1 to C17-P6)
// ------------------------------------------------------------------------------
MERGE (p1:Product {stockCode: "85123A"})
ON CREATE SET p1.description = "WHITE HANGING HEART T-LIGHT HOLDER",
              p1.unitPrice = 2.55,
              p1.posterAlias = "C17-P1"
MERGE (p1)-[:BELONGS_TO]->(c17);

MERGE (p2:Product {stockCode: "71053"})
ON CREATE SET p2.description = "WHITE METAL LANTERN",
              p2.unitPrice = 3.39,
              p2.posterAlias = "C17-P2"
MERGE (p2)-[:BELONGS_TO]->(c17);

MERGE (p3:Product {stockCode: "84406B"})
ON CREATE SET p3.description = "CREAM CUPID HEARTS COAT HANGER",
              p3.unitPrice = 2.75,
              p3.posterAlias = "C17-P3"
MERGE (p3)-[:BELONGS_TO]->(c17);

MERGE (p4:Product {stockCode: "22178"})
ON CREATE SET p4.description = "VICTORIAN GLASS HANGING T-LIGHT",
              p4.unitPrice = 1.25,
              p4.posterAlias = "C17-P4"
MERGE (p4)-[:BELONGS_TO]->(c17);

MERGE (p5:Product {stockCode: "82494L"})
ON CREATE SET p5.description = "WOODEN FRAME ANTIQUE WHITE",
              p5.unitPrice = 2.95,
              p5.posterAlias = "C17-P5"
MERGE (p5)-[:BELONGS_TO]->(c17);

MERGE (p6:Product {stockCode: "84879"})
ON CREATE SET p6.description = "ASSORTED COLOUR BIRD ORNAMENT",
              p6.unitPrice = 1.69,
              p6.posterAlias = "C17-P6"
MERGE (p6)-[:BELONGS_TO]->(c17);

MERGE (p_bag1:Product {stockCode: "20725"})
ON CREATE SET p_bag1.description = "LUNCH BAG RED RETROSPOT",
              p_bag1.unitPrice = 1.65,
              p_bag1.posterAlias = "C08-P1"
MERGE (p_bag1)-[:BELONGS_TO]->(c08);

MERGE (p_bag2:Product {stockCode: "85099B"})
ON CREATE SET p_bag2.description = "JUMBO BAG RED RETROSPOT",
              p_bag2.unitPrice = 1.95,
              p_bag2.posterAlias = "C08-P2"
MERGE (p_bag2)-[:BELONGS_TO]->(c08);

// ------------------------------------------------------------------------------
// 4. Customers
// ------------------------------------------------------------------------------

// Customer 13085 (Design Anchor Customer from Slides 1 & 3)
MERGE (cust1:Customer {id: 13085})
ON CREATE SET cust1.country = "United Kingdom",
              cust1.tenureTier = "Established",
              cust1.totalSpendPreCutoff = 2433.84,
              cust1.orderCountPreCutoff = 7,
              cust1.actual60DayRepurchase = 1,
              cust1.isDesignAnchor = true
MERGE (cust1)-[:SHARES_COMMUNITY]->(c17);

// Customer 17850 (Peer Customer in C17 Community)
MERGE (cust2:Customer {id: 17850})
ON CREATE SET cust2.country = "United Kingdom",
              cust2.tenureTier = "Established",
              cust2.totalSpendPreCutoff = 1820.50,
              cust2.orderCountPreCutoff = 5,
              cust2.actual60DayRepurchase = 1
MERGE (cust2)-[:SHARES_COMMUNITY]->(c17);

// Customer 19999 (Slide 7 Failure-Case Guardrail: 100% Cancelled Invoices)
MERGE (cust3:Customer {id: 19999})
ON CREATE SET cust3.country = "United Kingdom",
              cust3.tenureTier = "New",
              cust3.totalSpendPreCutoff = 0.0,
              cust3.orderCountPreCutoff = 2,
              cust3.isCancelledOnly = true,
              cust3.guardrailStatus = "Insufficient Evidence - 100% Cancelled Transactions",
              cust3.actual60DayRepurchase = 0;

// ------------------------------------------------------------------------------
// 5. Invoices and InvoiceLines (Concrete Evidence Provenance)
// ------------------------------------------------------------------------------

// Invoice 536365
MERGE (inv1:Invoice {id: "536365"})
ON CREATE SET inv1.date = "2010-12-01",
              inv1.isCancelled = false,
              inv1.totalAmount = 354.44;

MERGE (cust1)-[:PLACED]->(inv1);
MERGE (cust2)-[:PLACED]->(inv1);

// InvoiceLine items for Invoice 536365
MERGE (line1:InvoiceLine {id: "536365-85123A"})
ON CREATE SET line1.quantity = 48, line1.unitPrice = 2.55, line1.lineTotal = 122.40;
MERGE (inv1)-[:CONTAINS]->(line1);
MERGE (line1)-[:OF_PRODUCT]->(p1);

MERGE (line2:InvoiceLine {id: "536365-71053"})
ON CREATE SET line2.quantity = 36, line2.unitPrice = 3.39, line2.lineTotal = 122.04;
MERGE (inv1)-[:CONTAINS]->(line2);
MERGE (line2)-[:OF_PRODUCT]->(p2);

MERGE (line3:InvoiceLine {id: "536365-84406B"})
ON CREATE SET line3.quantity = 40, line3.unitPrice = 2.75, line3.lineTotal = 110.00;
MERGE (inv1)-[:CONTAINS]->(line3);
MERGE (line3)-[:OF_PRODUCT]->(p3);

// Invoice 536366
MERGE (inv2:Invoice {id: "536366"})
ON CREATE SET inv2.date = "2010-12-06",
              inv2.isCancelled = false,
              inv2.totalAmount = 177.60;
MERGE (cust1)-[:PLACED]->(inv2);

// Invoice 536372
MERGE (inv3:Invoice {id: "536372"})
ON CREATE SET inv3.date = "2011-03-31",
              inv3.isCancelled = false,
              inv3.totalAmount = 701.20;
MERGE (cust1)-[:PLACED]->(inv3);

// Bipartite Projected Customer-Product Relationships
MERGE (cust1)-[r1:PURCHASED]->(p1)
ON CREATE SET r1.totalSpend = 275.40, r1.totalQuantity = 108, r1.orderCount = 2;

MERGE (cust1)-[r2:PURCHASED]->(p2)
ON CREATE SET r2.totalSpend = 122.04, r2.totalQuantity = 36, r2.orderCount = 1;

MERGE (cust2)-[r3:PURCHASED]->(p1)
ON CREATE SET r3.totalSpend = 153.00, r3.totalQuantity = 60, r3.orderCount = 1;

// ------------------------------------------------------------------------------
// 6. Feature Vector (Customer 13085 Multimodal Profile)
// ------------------------------------------------------------------------------
MERGE (fv1:FeatureVector {id: "FEAT_13085"})
ON CREATE SET fv1.customerId = 13085,
              fv1.recencyDays = 62,
              fv1.frequencyInvoices = 7,
              fv1.monetarySpend = 2433.84,
              fv1.primaryCommunityId = "C17",
              fv1.graphDegree = 8,
              fv1.coPurchasingNeighbors = 14,
              fv1.predictedRepurchaseRisk = 0.88,
              fv1.targetRepurchaseLabel = 1;

MERGE (cust1)-[:HAS_FEATURE_VECTOR]->(fv1);

// ------------------------------------------------------------------------------
// 7. Experimental Results & Ablation (Model Evaluation Subgraph)
// ------------------------------------------------------------------------------
MERGE (exp1:Experiment {name: "Original GraphRAG"})
ON CREATE SET exp1.notes = "RFM + invoice paths only",
              exp1.prAuc = 0.31,
              exp1.recallAt50 = 0.42,
              exp1.faithfulness = 0.78,
              exp1.usesCommunityFeatures = false;

MERGE (exp2:Experiment {name: "Community-Enhanced GraphRAG"})
ON CREATE SET exp2.notes = "+ community features & multi-hop",
              exp2.prAuc = 0.39,
              exp2.recallAt50 = 0.51,
              exp2.faithfulness = 0.86,
              exp2.usesCommunityFeatures = true,
              exp2.prAucLift = 0.08,
              exp2.recallLift = 0.09,
              exp2.faithfulnessLift = 0.08;

MERGE (exp1)-[:ENHANCED_BY {
  prAucLift: 0.08, 
  recallLift: 0.09, 
  faithfulnessLift: 0.08, 
  signal: "Primary experimental signal"
}]->(exp2);

MERGE (exp2)-[:LEVERAGES_COMMUNITY]->(c17);
MERGE (exp2)-[:LEVERAGES_COMMUNITY]->(c08);

// ------------------------------------------------------------------------------
// 8. Agentic Tools & Lineage
// ------------------------------------------------------------------------------
MERGE (tool1:Tool {name: "Grok"})
ON CREATE SET tool1.role = "Primary agentic partner for conceptual design, prompt chaining & feature prioritization";

MERGE (tool2:Tool {name: "Neo4j Aura"})
ON CREATE SET tool2.role = "Cloud property graph database engine for zero-leakage ingestion & Cypher GraphRAG";

MERGE (tool3:Tool {name: "Google Antigravity"})
ON CREATE SET tool3.role = "Agentic pair programmer for automated code synthesis, validation & interactive visualization";

MERGE (exp2)-[:DESIGNED_WITH]->(tool1);
MERGE (exp2)-[:RUNS_ON]->(tool2);
MERGE (exp2)-[:IMPLEMENTED_WITH]->(tool3);

// ==============================================================================
// 9. Multi-Hop GraphRAG Retrieval & Evidence Queries
// ==============================================================================

// GraphRAG Query 1: Full Invoice Evidence Provenance for Customer 13085 Repurchase Risk
// Tracing from Customer -> Placed Invoices -> Line Items -> Products -> Community
MATCH (c:Customer {id: 13085})-[:PLACED]->(inv:Invoice)-[:CONTAINS]->(line:InvoiceLine)-[:OF_PRODUCT]->(p:Product)-[:BELONGS_TO]->(comm:Community)
RETURN c.id AS CustomerID,
       inv.id AS InvoiceID,
       inv.date AS InvoiceDate,
       p.stockCode AS StockCode,
       p.description AS ProductDescription,
       line.quantity AS Quantity,
       line.lineTotal AS LineTotal,
       comm.id AS CommunityID,
       comm.theme AS CommunityTheme
ORDER BY inv.date ASC, line.lineTotal DESC;

// GraphRAG Query 2: Look-Alike Peer Product Recommendations via Community C17
// Discovering products in C17 purchased by peers that Customer 13085 has not yet bought
MATCH (c:Customer {id: 13085})-[:SHARES_COMMUNITY]->(comm:Community)<-[:BELONGS_TO]-(p:Product)
WHERE NOT (c)-[:PURCHASED]->(p)
RETURN comm.id AS Community,
       comm.theme AS CommunityTheme,
       p.stockCode AS RecommendedStockCode,
       p.description AS ProductDescription,
       p.unitPrice AS Price,
       p.posterAlias AS PosterAlias;

// GraphRAG Query 3: Failure-Case Guardrail Audit (Detecting 100% Cancelled Customers)
MATCH (c:Customer)
WHERE c.isCancelledOnly = true
RETURN c.id AS FlaggedCustomer,
       c.guardrailStatus AS AuditWarning,
       "Do not generate predictive repurchase score without positive invoice history." AS ActionRequired;

// GraphRAG Query 4: Summary of Ablation Lift (+0.08 PR-AUC)
MATCH (base:Experiment {name: "Original GraphRAG"})-[rel:ENHANCED_BY]->(enhanced:Experiment)
RETURN base.name AS Baseline,
       base.prAuc AS BaselinePRAUC,
       enhanced.name AS EnhancedModel,
       enhanced.prAuc AS EnhancedPRAUC,
       rel.prAucLift AS DeltaLift,
       rel.faithfulnessLift AS FaithfulnessLift;
