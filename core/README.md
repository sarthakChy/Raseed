# Database Schemas for RASEED Multi-Agent System

## Firestore Collections Schema

### 1. Users Collection (`users/{userId}`)
```json
{
  "userId": "string",
  "email": "string",
  "displayName": "string",
  "photoURL": "string",
  "createdAt": "timestamp",
  "lastLoginAt": "timestamp",
  "preferences": {
    "currency": "USD",
    "language": "en",
    "timezone": "America/New_York",
    "notifications": {
      "pushEnabled": true,
      "emailEnabled": false,
      "budgetAlerts": true,
      "spendingInsights": true,
      "weeklyReports": false,
      "proactiveInsights": true,
      "frequency": "daily" // daily, weekly, monthly
    },
    "privacySettings": {
      "shareData": false,
      "anonymousAnalytics": true
    }
  },
  "financialProfile": {
    "monthlyIncome": 0,
    "budgetLimits": {
      "total": 0,
      "groceries": 0,
      "dining": 0,
      "entertainment": 0,
      "transportation": 0,
      "shopping": 0,
      "utilities": 0,
      "healthcare": 0,
      "other": 0
    },
    "financialGoals": [
      {
        "id": "string",
        "type": "saving" // saving, budget_reduction, debt_payoff
        "targetAmount": 0,
        "currentAmount": 0,
        "deadline": "timestamp",
        "category": "string",
        "priority": "high" // high, medium, low
      }
    ],
    "riskTolerance": "moderate" // conservative, moderate, aggressive
  }
}
```

### 2. Chat Sessions Collection (`chatSessions/{sessionId}`)
```json
{
  "sessionId": "string",
  "userId": "string",
  "startedAt": "timestamp",
  "lastActiveAt": "timestamp",
  "status": "active", // active, ended, archived
  "context": {
    "currentTopic": "spending_analysis", // spending_analysis, budgeting, recommendations
    "lastIntent": "ANALYTICAL",
    "conversationFlow": ["translate_query", "analyze_data"],
    "activeWorkflowId": "string"
  },
  "metadata": {
    "deviceType": "web", // web, mobile, api
    "userAgent": "string",
    "ipAddress": "string"
  }
}
```

### 3. Chat Messages Collection (`chatSessions/{sessionId}/messages/{messageId}`)
```json
{
  "messageId": "string",
  "sessionId": "string",
  "userId": "string",
  "timestamp": "timestamp",
  "type": "user", // user, agent, system
  "content": {
    "text": "string",
    "intent": "ANALYTICAL", // ANALYTICAL, EXPLORATORY, ACTIONABLE, COMPARATIVE, PREDICTIVE
    "entities": {
      "amount": 450.75,
      "category": "groceries",
      "timeframe": "last_month",
      "merchants": ["Whole Foods"]
    },
    "attachments": [
      {
        "type": "chart", // chart, image, receipt
        "url": "string",
        "metadata": {}
      }
    ]
  },
  "agentContext": {
    "agentId": "financial_analysis_agent",
    "workflowId": "string",
    "stepId": "analyze_data",
    "processingTime": 1250, // milliseconds
    "confidence": 0.95
  },
  "isRead": true,
  "reactions": {
    "helpful": false,
    "notHelpful": false
  }
}
```

### 4. Receipt Processing Queue (`receiptQueue/{receiptId}`)
```json
{
  "receiptId": "string",
  "userId": "string",
  "uploadedAt": "timestamp",
  "processedAt": "timestamp",
  "status": "pending", // pending, processing, completed, failed
  "ocrData": {
    "rawText": "string",
    "confidence": 0.89,
    "extractedData": {
      "merchantName": "WHOLE FOODS MKT #123",
      "normalizedMerchant": "Whole Foods",
      "date": "2024-01-15",
      "totalAmount": 127.83,
      "subtotal": 119.84,
      "tax": 7.99,
      "items": [
        {
          "name": "Organic Quinoa",
          "price": 8.99,
          "quantity": 1,
          "category": "grains",
          "isOrganic": true
        }
      ],
      "paymentMethod": "Credit Card",
      "address": {
        "street": "123 Main St",
        "city": "Austin",
        "state": "TX",
        "zipCode": "78701"
      }
    }
  },
  "processingErrors": [],
  "retryCount": 0,
  "walletPass": {
    "objectId": "string",
    "walletLink": "string",
    "created": false
  }
}
```

### 5. Notifications Collection (`notifications/{notificationId}`)
```json
{
  "notificationId": "string",
  "userId": "string",
  "type": "proactive_insight", // proactive_insight, budget_alert, goal_update, system
  "title": "Unusual Spending Pattern Detected",
  "message": "You've spent 30% more on dining out this week compared to your average.",
  "data": {
    "category": "dining",
    "amount": 285.50,
    "comparison": "weekly_average",
    "variance": 0.30,
    "actionable": true,
    "recommendations": ["dining_alternatives", "budget_adjustment"]
  },
  "priority": "medium", // low, medium, high, urgent
  "status": "pending", // pending, sent, read, dismissed
  "scheduledFor": "timestamp",
  "sentAt": "timestamp",
  "readAt": "timestamp",
  "channels": ["push", "in_app"], // push, email, in_app, sms
  "metadata": {
    "source": "proactive_insights_agent",
    "triggerEvent": "spending_pattern_analysis",
    "relevanceScore": 0.87
  }
}
```

---

## PostgreSQL Schema

### Core Tables

#### 1. Users Table
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    firebase_uid VARCHAR(128) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    
    -- Financial Profile
    monthly_income DECIMAL(12,2) DEFAULT 0,
    risk_tolerance VARCHAR(20) DEFAULT 'moderate',
    
    -- Preferences stored as JSONB for flexibility
    preferences JSONB DEFAULT '{}',
    
    -- User behavioral patterns (for ML)
    spending_patterns JSONB DEFAULT '{}',
    
    -- User preference embedding for personalization
    preference_embedding VECTOR(768),
    
    CONSTRAINT valid_risk_tolerance CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive'))
);

-- Indexes
CREATE INDEX idx_users_firebase_uid ON users(firebase_uid);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;
```

#### 2. Merchants Table
```sql
CREATE TABLE merchants (
    merchant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    
    -- Location information
    address JSONB,
    coordinates POINT, -- For geographic queries
    
    -- Merchant characteristics for recommendations
    price_range VARCHAR(20), -- budget, mid-range, premium, luxury
    merchant_type VARCHAR(50), -- chain, local, online, etc.
    
    -- Merchant embedding for similarity searches
    merchant_embedding VECTOR(768),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Aggregated data for quick access
    avg_transaction_amount DECIMAL(12,2),
    total_transactions INTEGER DEFAULT 0,
    user_count INTEGER DEFAULT 0
);

-- Indexes
CREATE UNIQUE INDEX idx_merchants_normalized ON merchants(normalized_name);
CREATE INDEX idx_merchants_category ON merchants(category);
CREATE INDEX idx_merchants_location ON merchants USING GIST(coordinates);
CREATE INDEX idx_merchants_embedding ON merchants USING ivfflat(merchant_embedding vector_cosine_ops);
```

#### 3. Transactions Table
```sql
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    merchant_id UUID REFERENCES merchants(merchant_id),
    
    -- Basic transaction info
    amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    transaction_date DATE NOT NULL,
    transaction_time TIME,
    
    -- Categorization
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    tags TEXT[],
    
    -- Payment details
    payment_method VARCHAR(50),
    card_last_four VARCHAR(4),
    
    -- Receipt information
    receipt_id UUID,
    receipt_image_url TEXT,
    
    -- Processing metadata
    confidence_score DECIMAL(3,2), -- OCR/processing confidence
    is_verified BOOLEAN DEFAULT false,
    verification_method VARCHAR(50), -- manual, auto, ml
    
    -- Transaction context embedding for pattern recognition
    transaction_embedding VECTOR(768),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Soft delete
    deleted_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT positive_amount CHECK (amount > 0),
    CONSTRAINT valid_confidence CHECK (confidence_score BETWEEN 0 AND 1)
);

-- Indexes for optimal query performance
CREATE INDEX idx_transactions_user_date ON transactions(user_id, transaction_date DESC);
CREATE INDEX idx_transactions_category ON transactions(category);
CREATE INDEX idx_transactions_merchant ON transactions(merchant_id);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_transactions_embedding ON transactions USING ivfflat(transaction_embedding vector_cosine_ops);
CREATE INDEX idx_transactions_active ON transactions(user_id, deleted_at) WHERE deleted_at IS NULL;

-- Composite indexes for common query patterns
CREATE INDEX idx_transactions_user_category_date ON transactions(user_id, category, transaction_date DESC);
CREATE INDEX idx_transactions_user_merchant_date ON transactions(user_id, merchant_id, transaction_date DESC);
```

#### 4. Transaction Items Table
```sql
CREATE TABLE transaction_items (
    item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    
    -- Item details
    name VARCHAR(255) NOT NULL,
    description TEXT,
    quantity DECIMAL(8,2) DEFAULT 1,
    unit_price DECIMAL(12,2) NOT NULL,
    total_price DECIMAL(12,2) NOT NULL,
    
    -- Categorization
    category VARCHAR(100),
    subcategory VARCHAR(100),
    
    -- Item attributes
    brand VARCHAR(100),
    size_info VARCHAR(100),
    is_organic BOOLEAN DEFAULT false,
    is_sale_item BOOLEAN DEFAULT false,
    discount_amount DECIMAL(12,2) DEFAULT 0,
    
    -- Item embedding for product similarity and substitution recommendations
    item_embedding VECTOR(768),
    
    -- Nutritional/product data (stored as JSONB for flexibility)
    product_metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT positive_quantity CHECK (quantity > 0),
    CONSTRAINT positive_unit_price CHECK (unit_price > 0),
    CONSTRAINT valid_total_price CHECK (total_price = quantity * unit_price - discount_amount)
);

-- Indexes
CREATE INDEX idx_transaction_items_transaction ON transaction_items(transaction_id);
CREATE INDEX idx_transaction_items_category ON transaction_items(category);
CREATE INDEX idx_transaction_items_brand ON transaction_items(brand);
CREATE INDEX idx_transaction_items_embedding ON transaction_items USING ivfflat(item_embedding vector_cosine_ops);
```

#### 5. Budget Limits Table
```sql
CREATE TABLE budget_limits (
    budget_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    category VARCHAR(100) NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- monthly, weekly, yearly
    limit_amount DECIMAL(12,2) NOT NULL,
    
    -- Time validity
    effective_from DATE NOT NULL,
    effective_to DATE,
    
    -- Tracking
    current_spent DECIMAL(12,2) DEFAULT 0,
    last_calculated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT positive_limit CHECK (limit_amount > 0),
    CONSTRAINT valid_period CHECK (period_type IN ('daily', 'weekly', 'monthly', 'yearly')),
    CONSTRAINT valid_date_range CHECK (effective_to IS NULL OR effective_to > effective_from)
);

-- Indexes
CREATE UNIQUE INDEX idx_budget_limits_unique ON budget_limits(user_id, category, period_type, effective_from);
CREATE INDEX idx_budget_limits_active ON budget_limits(user_id, effective_from, effective_to);
```

#### 6. Financial Goals Table
```sql
CREATE TABLE financial_goals (
    goal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    title VARCHAR(255) NOT NULL,
    description TEXT,
    goal_type VARCHAR(50) NOT NULL, -- saving, budget_reduction, debt_payoff, investment
    
    -- Financial targets
    target_amount DECIMAL(12,2),
    current_amount DECIMAL(12,2) DEFAULT 0,
    target_date DATE,
    
    -- Categorization
    category VARCHAR(100), -- Which spending category this relates to
    priority VARCHAR(20) DEFAULT 'medium', -- low, medium, high
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'active', -- active, completed, paused, cancelled
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    
    -- Goal metadata
    milestones JSONB DEFAULT '[]',
    strategies JSONB DEFAULT '[]', -- Recommended strategies to achieve goal
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_goal_type CHECK (goal_type IN ('saving', 'budget_reduction', 'debt_payoff', 'investment')),
    CONSTRAINT valid_priority CHECK (priority IN ('low', 'medium', 'high')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'completed', 'paused', 'cancelled')),
    CONSTRAINT valid_progress CHECK (progress_percentage BETWEEN 0 AND 100)
);

-- Indexes
CREATE INDEX idx_financial_goals_user ON financial_goals(user_id);
CREATE INDEX idx_financial_goals_status ON financial_goals(user_id, status);
CREATE INDEX idx_financial_goals_priority ON financial_goals(user_id, priority);
```

### Analysis & Intelligence Tables

#### 7. Spending Patterns Table
```sql
CREATE TABLE spending_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    pattern_type VARCHAR(50) NOT NULL, -- weekly_cycle, monthly_trend, seasonal, merchant_loyalty, etc.
    category VARCHAR(100),
    
    -- Pattern characteristics
    frequency VARCHAR(20), -- daily, weekly, monthly, quarterly, yearly
    confidence_score DECIMAL(3,2) NOT NULL,
    
    -- Pattern data
    pattern_data JSONB NOT NULL, -- Stores the actual pattern metrics
    statistical_summary JSONB, -- Mean, median, std dev, etc.
    
    -- Time range this pattern was detected over
    analysis_start_date DATE NOT NULL,
    analysis_end_date DATE NOT NULL,
    
    -- Pattern validity
    is_active BOOLEAN DEFAULT true,
    last_validated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_confidence CHECK (confidence_score BETWEEN 0 AND 1)
);

-- Indexes
CREATE INDEX idx_spending_patterns_user ON spending_patterns(user_id);
CREATE INDEX idx_spending_patterns_type ON spending_patterns(pattern_type);
CREATE INDEX idx_spending_patterns_active ON spending_patterns(user_id, is_active) WHERE is_active = true;
```

#### 8. Insights Table
```sql
CREATE TABLE insights (
    insight_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    insight_type VARCHAR(50) NOT NULL, -- spending_spike, budget_warning, goal_progress, recommendation
    category VARCHAR(100),
    
    -- Insight content
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    
    -- Insight metadata
    severity VARCHAR(20) DEFAULT 'medium', -- low, medium, high, critical
    confidence_score DECIMAL(3,2) NOT NULL,
    relevance_score DECIMAL(3,2) NOT NULL,
    
    -- Supporting data
    data_points JSONB, -- The data that generated this insight
    recommendations JSONB, -- Related recommendations
    
    -- Insight lifecycle
    status VARCHAR(20) DEFAULT 'active', -- active, dismissed, acted_upon, expired
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- User interaction
    viewed_at TIMESTAMP WITH TIME ZONE,
    dismissed_at TIMESTAMP WITH TIME ZONE,
    acted_upon_at TIMESTAMP WITH TIME ZONE,
    
    -- Generation metadata
    generated_by VARCHAR(50), -- Which agent generated this
    generation_context JSONB, -- Context data used for generation
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_severity CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'dismissed', 'acted_upon', 'expired')),
    CONSTRAINT valid_confidence CHECK (confidence_score BETWEEN 0 AND 1),
    CONSTRAINT valid_relevance CHECK (relevance_score BETWEEN 0 AND 1)
);

-- Indexes
CREATE INDEX idx_insights_user_status ON insights(user_id, status);
CREATE INDEX idx_insights_user_created ON insights(user_id, created_at DESC);
CREATE INDEX idx_insights_severity ON insights(user_id, severity) WHERE status = 'active';
```

#### 9. Recommendations Table
```sql
CREATE TABLE recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    insight_id UUID REFERENCES insights(insight_id),
    
    recommendation_type VARCHAR(50) NOT NULL, -- alternative_merchant, budget_adjustment, goal_creation, etc.
    category VARCHAR(100),
    
    -- Recommendation content
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    action_text VARCHAR(255), -- What the user should do
    
    -- Recommendation data
    potential_savings DECIMAL(12,2),
    effort_level VARCHAR(20), -- low, medium, high
    estimated_impact VARCHAR(20), -- low, medium, high
    
    -- Alternative recommendations (for merchant suggestions, etc.)
    alternatives JSONB, -- Array of alternative options
    
    -- Recommendation scoring
    relevance_score DECIMAL(3,2) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    priority_score DECIMAL(3,2) NOT NULL,
    
    -- User interaction
    status VARCHAR(20) DEFAULT 'pending', -- pending, accepted, rejected, expired
    viewed_at TIMESTAMP WITH TIME ZONE,
    responded_at TIMESTAMP WITH TIME ZONE,
    response_feedback TEXT,
    
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_effort CHECK (effort_level IN ('low', 'medium', 'high')),
    CONSTRAINT valid_impact CHECK (estimated_impact IN ('low', 'medium', 'high')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'accepted', 'rejected', 'expired')),
    CONSTRAINT valid_scores CHECK (
        relevance_score BETWEEN 0 AND 1 AND 
        confidence_score BETWEEN 0 AND 1 AND 
        priority_score BETWEEN 0 AND 1
    )
);

-- Indexes
CREATE INDEX idx_recommendations_user_status ON recommendations(user_id, status);
CREATE INDEX idx_recommendations_priority ON recommendations(user_id, priority_score DESC) WHERE status = 'pending';
```

#### 10. Chat Query Embeddings Table
```sql
CREATE TABLE chat_queries (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id VARCHAR(255), -- References Firestore session
    
    -- Query content
    original_query TEXT NOT NULL,
    processed_query TEXT,
    intent_type VARCHAR(50) NOT NULL,
    
    -- Query classification
    entities JSONB, -- Extracted entities (amounts, dates, categories, etc.)
    confidence_score DECIMAL(3,2),
    
    -- Query embedding for semantic search and similar query matching
    query_embedding VECTOR(768),
    
    -- Query context
    conversation_context JSONB, -- Previous queries in conversation
    user_context JSONB, -- User's current financial state when query was made
    
    -- Response tracking
    response_provided BOOLEAN DEFAULT false,
    response_quality_score DECIMAL(3,2), -- If user provides feedback
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_intent CHECK (intent_type IN ('ANALYTICAL', 'EXPLORATORY', 'ACTIONABLE', 'COMPARATIVE', 'PREDICTIVE')),
    CONSTRAINT valid_confidence CHECK (confidence_score BETWEEN 0 AND 1),
    CONSTRAINT valid_quality CHECK (response_quality_score IS NULL OR response_quality_score BETWEEN 0 AND 1)
);

-- Indexes
CREATE INDEX idx_chat_queries_user ON chat_queries(user_id, created_at DESC);
CREATE INDEX idx_chat_queries_intent ON chat_queries(intent_type);
CREATE INDEX idx_chat_queries_embedding ON chat_queries USING ivfflat(query_embedding vector_cosine_ops);
```

#### 11. Google Wallet Passes Table
```sql
CREATE TABLE wallet_passes (
    pass_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    transaction_id UUID REFERENCES transactions(transaction_id),
    
    -- Google Wallet identifiers
    object_id VARCHAR(255) UNIQUE NOT NULL,
    class_id VARCHAR(255) NOT NULL,
    
    -- Pass details
    wallet_link TEXT NOT NULL,
    pass_status VARCHAR(20) DEFAULT 'active', -- active, expired, revoked
    
    -- Pass metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_accessed TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    
    -- Pass data snapshot (for analytics)
    pass_data JSONB,
    
    CONSTRAINT valid_pass_status CHECK (pass_status IN ('active', 'expired', 'revoked'))
);

-- Indexes
CREATE INDEX idx_wallet_passes_user ON wallet_passes(user_id);
CREATE INDEX idx_wallet_passes_transaction ON wallet_passes(transaction_id);
CREATE UNIQUE INDEX idx_wallet_passes_object_id ON wallet_passes(object_id);
```

### Utility Functions and Triggers

```sql
-- Function to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_merchants_updated_at BEFORE UPDATE ON merchants FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_budget_limits_updated_at BEFORE UPDATE ON budget_limits FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_financial_goals_updated_at BEFORE UPDATE ON financial_goals FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate goal progress
CREATE OR REPLACE FUNCTION calculate_goal_progress()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.target_amount > 0 THEN
        NEW.progress_percentage = (NEW.current_amount / NEW.target_amount) * 100;
        IF NEW.progress_percentage >= 100 AND NEW.status = 'active' THEN
            NEW.status = 'completed';
            NEW.completed_at = CURRENT_TIMESTAMP;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_financial_goals_progress 
    BEFORE UPDATE ON financial_goals 
    FOR EACH ROW EXECUTE FUNCTION calculate_goal_progress();
```

## Key Design Decisions & Benefits

### 1. **Hybrid Storage Strategy**
- **Firestore**: Real-time chat, user sessions, immediate receipt processing queue
- **PostgreSQL**: All processed financial data, embeddings, analytics, and historical patterns

### 2. **Vector Embeddings Strategy**
- **User Preference Embeddings**: Enable personalized recommendations
- **Merchant Embeddings**: Find similar stores and alternatives
- **Item Embeddings**: Product substitution and alternative suggestions
- **Transaction Embeddings**: Pattern recognition and anomaly detection
- **Query Embeddings**: Semantic search for similar questions and better intent understanding

### 3. **Scalability Features**
- Partitioning on user_id for large tables
- Optimized indexes for common query patterns
- JSONB for flexible metadata storage
- Vector indexes using IVFFlat for efficient similarity searches

### 4. **Data Quality & Integrity**
- Comprehensive constraints and validations
- Soft deletes for important data
- Confidence scores for ML-generated content
- Audit trails with created_at/updated_at timestamps

This schema supports all your multi-agent workflows while maintaining performance and scalability for real-time financial analysis and personalized recommendations.