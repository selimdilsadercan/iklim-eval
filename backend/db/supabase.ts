import { createClient } from "@supabase/supabase-js";
import { secret } from "encore.dev/config";

// Supabase bağlantı bilgileri (Encore Secrets üzerinden güvenli erişim)
const SUPABASE_URL = secret("SupabaseUrl");
const SUPABASE_SERVICE_ROLE_KEY = secret("SupabaseServiceRoleKey");

/**
 * Supabase istemcisini başlatır.
 * Service Role Key kullanıyoruz çünkü backend tarafında tüm yetkilere (bypass RLS) ihtiyacımız var.
 */
export const supabase = createClient(
  SUPABASE_URL(),
  SUPABASE_SERVICE_ROLE_KEY(),
);

// Alternatif: Sadece okuma/yazma yetkili Anon Key lazımsa buraya eklenebilir.
