-- Migration: Create student reviews system tables
-- Created: 2024-01-19

-- Create course_reviews table
CREATE TABLE IF NOT EXISTS course_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    title TEXT,
    body TEXT NOT NULL,
    is_anonymous BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_edited BOOLEAN DEFAULT FALSE,
    helpful_count INTEGER DEFAULT 0,
    UNIQUE(course_id, user_id)
);

-- Create review_responses table for instructor responses
CREATE TABLE IF NOT EXISTS review_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES course_reviews(id) ON DELETE CASCADE,
    instructor_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    response TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create review_helpfulness table for helpful votes
CREATE TABLE IF NOT EXISTS review_helpfulness (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES course_reviews(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    is_helpful BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(review_id, user_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_course_reviews_course_id ON course_reviews(course_id);
CREATE INDEX IF NOT EXISTS idx_course_reviews_user_id ON course_reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_course_reviews_rating ON course_reviews(rating);
CREATE INDEX IF NOT EXISTS idx_course_reviews_created_at ON course_reviews(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_review_responses_review_id ON review_responses(review_id);
CREATE INDEX IF NOT EXISTS idx_review_helpfulness_review_id ON review_helpfulness(review_id);

-- Function to update course rating statistics
CREATE OR REPLACE FUNCTION update_course_rating_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the courses table with new rating statistics
    UPDATE courses 
    SET 
        rating_avg = (
            SELECT COALESCE(AVG(rating::DECIMAL), 0)
            FROM course_reviews 
            WHERE course_id = COALESCE(NEW.course_id, OLD.course_id)
        ),
        reviews_count = (
            SELECT COUNT(*)
            FROM course_reviews 
            WHERE course_id = COALESCE(NEW.course_id, OLD.course_id)
        )
    WHERE id = COALESCE(NEW.course_id, OLD.course_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create triggers to automatically update course statistics
DROP TRIGGER IF EXISTS trigger_update_course_rating_stats ON course_reviews;
CREATE TRIGGER trigger_update_course_rating_stats
    AFTER INSERT OR UPDATE OR DELETE ON course_reviews
    FOR EACH ROW
    EXECUTE FUNCTION update_course_rating_stats();

-- Function to update helpful count
CREATE OR REPLACE FUNCTION update_review_helpful_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE course_reviews 
    SET helpful_count = (
        SELECT COUNT(*)
        FROM review_helpfulness 
        WHERE review_id = COALESCE(NEW.review_id, OLD.review_id) 
        AND is_helpful = true
    )
    WHERE id = COALESCE(NEW.review_id, OLD.review_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update helpful count
DROP TRIGGER IF EXISTS trigger_update_review_helpful_count ON review_helpfulness;
CREATE TRIGGER trigger_update_review_helpful_count
    AFTER INSERT OR UPDATE OR DELETE ON review_helpfulness
    FOR EACH ROW
    EXECUTE FUNCTION update_review_helpful_count();

-- Add RLS (Row Level Security) policies
ALTER TABLE course_reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE review_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE review_helpfulness ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read all reviews
CREATE POLICY "Anyone can read reviews" ON course_reviews
    FOR SELECT USING (true);

-- Policy: Users can only insert their own reviews
CREATE POLICY "Users can insert own reviews" ON course_reviews
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy: Users can only update their own reviews
CREATE POLICY "Users can update own reviews" ON course_reviews
    FOR UPDATE USING (auth.uid() = user_id);

-- Policy: Users can only delete their own reviews
CREATE POLICY "Users can delete own reviews" ON course_reviews
    FOR DELETE USING (auth.uid() = user_id);

-- Policy: Anyone can read review responses
CREATE POLICY "Anyone can read review responses" ON review_responses
    FOR SELECT USING (true);

-- Policy: Only course instructors can respond to reviews
CREATE POLICY "Instructors can respond to reviews" ON review_responses
    FOR INSERT WITH CHECK (
        auth.uid() = instructor_id AND
        EXISTS (
            SELECT 1 FROM courses c
            JOIN course_reviews cr ON c.id = cr.course_id
            WHERE cr.id = review_id AND c.user_id = auth.uid()
        )
    );

-- Policy: Anyone can read helpfulness votes
CREATE POLICY "Anyone can read helpfulness" ON review_helpfulness
    FOR SELECT USING (true);

-- Policy: Users can manage their own helpfulness votes
CREATE POLICY "Users can manage own helpfulness votes" ON review_helpfulness
    FOR ALL USING (auth.uid() = user_id);